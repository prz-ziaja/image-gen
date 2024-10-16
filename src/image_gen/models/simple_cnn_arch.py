import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import image_gen.models.utils.blocks as blocks
import image_gen.models.utils.embeddings as embeddings
import image_gen.models.utils.image_diffusers as image_diffusers

import mlflow

def temp_vis(image):
    # TODO: rewrite this
    reverse_dataset_preprocessing = transforms.Compose([
        transforms.Lambda(lambda t: (torch.tensor([58.4, 57.12, 57.38]).reshape([3,1,1])*t + torch.tensor([123.68, 116.28, 103.53]).reshape([3,1,1]))/255),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_dataset_preprocessing(image[0].detach().cpu()))

class Model(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        IMG_CH = 3
        IMG_SIZE = config["image_size"][0]
        down_chs = (32, 64, 128)
        up_chs = down_chs[::-1]  # Reverse of the down channels
        latent_image_size = IMG_SIZE // 4 # 2 ** (len(down_chs) - 1)
        self.config = config

        self.image_diffuser = image_diffusers.imageDiffuser(config["T"])
        self.time_embedding_0 = embeddings.embedBlock(1, [16, up_chs[0]])
        self.time_embedding_1 = embeddings.embedBlock(1, [16, up_chs[1]])

        self._down = blocks.geluConv2d(IMG_CH, down_chs[0], stride=1)
        self._up = blocks.geluConv2d(up_chs[-1]*2, IMG_CH, stride=1)

        # Downsample
        self.down1 = blocks.downBlock2d(down_chs[0], down_chs[1], group_size=8) # New
        self.down2 = blocks.downBlock2d(down_chs[1], down_chs[2], group_size=8) # New
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2]*latent_image_size**2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[2]*latent_image_size**2),
            nn.ReLU()
        )

        # Upsample
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),
            blocks.geluConv2d(up_chs[0], up_chs[0], num_groups=8) # New
        )
        self.up1 = blocks.upBlock2d(up_chs[0], up_chs[1], 8) # New
        self.up2 = blocks.upBlock2d(up_chs[1], up_chs[2], 8) # New

        print(f"Device: {self.device}")
        self.lr = config["lr"]
        self.loss_function = config["loss_function"]

    def compute_loss(self, imgs, imgs_pred):
        return self.loss_function(imgs, imgs_pred)

    def forward(self, x, t:int):
        x = self._down(x)
        down0 = self.down1(x)
        down1 = self.down2(down0)
        latent_vec = self.to_vec(down1)

        latent_vec = self.dense_emb(latent_vec)
        t = t.float() / self.config["T"]  # Convert from [0, T] to [0, 1]
        temb_1 = self.time_embedding_0(t)
        temb_2 = self.time_embedding_1(t)

        up0 = self.up0(latent_vec)
        up1 = self.up1(up0+temb_1, down1)
        up2 = self.up2(up1+temb_2, down0)
        return self._up(torch.cat((up2, x), 1)) # New
    # def on_before_optimizer_step(self, optimizer) -> None:
    #     print("on_before_opt enter")
    #     for name,p in self.named_parameters():
    #         if p.grad is None:
    #             print(name, p.shape)

    #     print("on_before_opt exit")
    def training_step(self, train_batch, batch_idx):

        self.last_batch = train_batch["image"]

        x = train_batch["image"].to(self.device)
        t = torch.randint(0, self.config["T"], (x.shape[0],1), device=self.device)

        imgs_noisy, noise = self.image_diffuser(x, t)
        preds = self.forward(imgs_noisy, t)
        loss = self.compute_loss(noise, preds)

        self.log("train_loss", loss.detach().cpu().item())
        # print(f"Recent loss: {loss.detach().cpu().item():.5f}", end="\r")
        if batch_idx % 25 == 0:
            self.on_train_epoch_end()
        return loss

    def on_train_epoch_end(self):
        filename = self.plot_sample(self.last_batch)
        mlflow.log_artifact(filename)

    @torch.no_grad()
    def plot_sample(self, imgs):
        # Take first image of batch
        imgs = imgs[[0], :, :, :]
        img, _ = self.image_diffuser(imgs[[0], :, :, :],19)
        org = torch.clone(img)
        for i in range(19, -1,-1):
            _t = torch.tensor([[i]],device=img.device)
            img = self.image_diffuser.reverse(img, self.forward(img, _t),_t)
            
            if i == 10:
                halfway = torch.clone(img)

        nrows = 2
        ncols = 2
        samples = {
            "Original" : imgs,
            "At the beginning": org,
            "Halfway" : halfway,
            "Predicted Original" : img
        }
        for i, (title, img) in enumerate(samples.items()):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.set_title(title)
            temp_vis(img)

        filename = f"/tmp/{int(time.time())}.png"
        plt.savefig(filename)
        return filename
        

    def validation_step(self, val_batch, batch_idx):
        return
        x, y = val_batch["hsv"], val_batch["labels"].flatten().long()

        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        self.eval_loss.append(loss)

        self.eval_preds.append(logits.argmax(dim=-1))
        self.eval_true.append(y)
        

    def on_validation_epoch_end(self):
        mlflow.pytorch.log_model(self, f"{int(time.time())}.pt")
        return
        avg_loss = torch.stack(self.eval_loss).cpu().mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.eval_loss.clear()

        f1_score = -classification_report(
            torch.cat(self.eval_true).cpu(),
            torch.cat(self.eval_preds).cpu(),
            output_dict=True
        )["weighted avg"]["f1-score"]
        self.log("val_neg_f1", f1_score, sync_dist=True)
        self.eval_preds.clear()
        self.eval_true.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
