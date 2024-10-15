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



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))

class Model(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        IMG_CH = 3
        IMG_SIZE = 64
        down_chs = (16, 32, 64)
        up_chs = down_chs[::-1]  # Reverse of the down channels
        latent_image_size = IMG_SIZE // 4 # 2 ** (len(down_chs) - 1)
        self.config = config
        self.config["T"] = 20

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
        #self.loss_function = config["loss_function"]#nn.CrossEntropyLoss()#HingeEmbeddingLoss()

    def compute_loss(self, imgs, imgs_pred):
        return F.mse_loss(imgs, imgs_pred)

    def forward(self, x, t:int):
        x = self._down(x)
        print(x.shape)
        down0 = self.down1(x)
        print(down0.shape)
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
        imgs_noisy = add_noise(x)
        preds = self.forward(imgs_noisy)
        loss = self.compute_loss(x, preds)

        self.log("train_loss", loss.detach().cpu().item())
        print(f"Recent loss: {loss.detach().cpu().item():.5f}", end="\r")
        if batch_idx % 25 == 0:
            self.on_train_epoch_end()
        return loss

    def on_train_epoch_end(self):
        filename = self.plot_sample(self.last_batch)
        run_id = self.logger.run_id
        self.logger.experiment.log_artifact(run_id=run_id, local_path=filename)

    @torch.no_grad()
    def plot_sample(self, imgs):
        # Take first image of batch
        imgs = imgs[[0], :, :, :]
        imgs_noisy = add_noise(imgs[[0], :, :, :])
        imgs_pred = self.forward(imgs_noisy)

        nrows = 1
        ncols = 3
        samples = {
            "Original" : imgs,
            "Noise Added" : imgs_noisy,
            "Predicted Original" : imgs_pred
        }
        for i, (title, img) in enumerate(samples.items()):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.set_title(title)
            show_tensor_image(img)

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
