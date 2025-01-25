import lightning.pytorch as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.transforms as transforms
import image_gen.models.utils.blocks as blocks
import image_gen.models.utils.embeddings as embeddings
import image_gen.models.utils.image_diffusers as image_diffusers

import mlflow


def temp_vis(image):
    # TODO: rewrite this
    reverse_dataset_preprocessing = transforms.Compose(
        [
            # transforms.Lambda(lambda t: (torch.tensor([58.4, 57.12, 57.38]).reshape([3,1,1])*t + torch.tensor([123.68, 116.28, 103.53]).reshape([3,1,1]))/255),
            # transforms.Lambda(lambda t: torch.minimum(torch.tensor([255]), t)),
            # transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
            # transforms.Lambda(lambda t: t.to(torch.uint8)),
            transforms.Lambda(lambda t: t - t.min()),
            transforms.Lambda(lambda t: t / (t.max() + 0.0001)),
        ]
    )

    plt.imshow(
        reverse_dataset_preprocessing(image[0].detach().cpu()).permute([1, 2, 0])
    )


class Model(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        IMG_CH = config["image_ch"]
        IMG_SIZE = config["image_size"][0]
        ENCODED_SENTENCE_SIZE = config["encoded_sentence_size"]
        down_chs = (64, 64, 128, 128, 128)
        up_chs = down_chs[::-1]  # Reverse of the down channels
        latent_image_size = IMG_SIZE // (2 ** (len(down_chs) - 1))
        self.config = config

        self.image_diffuser = image_diffusers.imageDiffuser(
            config["T"], config["t_start"], config["t_end"]
        )

        self.time_embedding_0 = embeddings.embedBlock(1, [16, up_chs[0]])
        self.time_embedding_1 = embeddings.embedBlock(1, [16, up_chs[1]])
        self.time_embedding_2 = embeddings.embedBlock(1, [16, up_chs[2]])
        self.time_embedding_3 = embeddings.embedBlock(1, [16, up_chs[3]])

        self.cat_embedding_0 = embeddings.embedBlock(
            ENCODED_SENTENCE_SIZE, [16, up_chs[0]]
        )
        self.cat_embedding_1 = embeddings.embedBlock(
            ENCODED_SENTENCE_SIZE, [16, up_chs[1]]
        )
        self.cat_embedding_2 = embeddings.embedBlock(
            ENCODED_SENTENCE_SIZE, [16, up_chs[2]]
        )
        self.cat_embedding_3 = embeddings.embedBlock(
            ENCODED_SENTENCE_SIZE, [16, up_chs[3]]
        )

        self._down = blocks.activatedConv2d(IMG_CH, down_chs[0], stride=1)
        self._up = nn.Sequential(
            nn.Conv2d(2 * up_chs[-1], up_chs[-1], 3, 1, 1),
            nn.GroupNorm(8, up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], IMG_CH, 3, 1, 1),
        )

        # Downsample
        self.down1 = blocks.downBlock2d(down_chs[0], down_chs[1], group_size=8)  # New
        self.down2 = blocks.downBlock2d(down_chs[1], down_chs[2], group_size=8)  # New
        self.down3 = blocks.downBlock2d(down_chs[2], down_chs[3], group_size=8)  # New
        self.down4 = blocks.downBlock2d(down_chs[3], down_chs[4], group_size=8)  # New
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[-1] * latent_image_size**2, down_chs[-2]),
            nn.ReLU(),
            nn.Linear(down_chs[-2], down_chs[-2]),
            nn.ReLU(),
            nn.Linear(down_chs[-2], down_chs[-1] * latent_image_size**2),
            nn.ReLU(),
        )

        # Upsample
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),
            blocks.activatedConv2d(up_chs[0], up_chs[0], num_groups=8),  # New
        )
        self.up1 = blocks.upBlock2dDoubleInput(
            up_chs[0], up_chs[0], up_chs[1], group_size=8
        )  # New
        self.up2 = blocks.upBlock2dDoubleInput(
            up_chs[1], up_chs[1], up_chs[2], group_size=8
        )  # New
        self.up3 = blocks.upBlock2dDoubleInput(
            up_chs[2], up_chs[2], up_chs[3], group_size=8
        )  # New
        self.up4 = blocks.upBlock2dDoubleInput(
            up_chs[3], up_chs[3], up_chs[4], group_size=8
        )  # New

        print(f"Device: {self.device}")
        self.lr = config["lr"]
        self.loss_function = config["loss_function"]
        self.cat_drop_prob = 0.1

    def compute_loss(self, imgs, imgs_pred):
        return self.loss_function(imgs, imgs_pred)

    def forward(self, x, t: int, cat: int):
        x = self._down(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        latent_vec = self.to_vec(down4)

        latent_vec = self.dense_emb(latent_vec)
        t = t.float() / self.config["T"]  # Convert from [0, T] to [0, 1]

        temb_1 = self.time_embedding_0(t)
        temb_2 = self.time_embedding_1(t)
        temb_3 = self.time_embedding_2(t)
        temb_4 = self.time_embedding_3(t)

        cat_1 = self.cat_embedding_0(cat)
        cat_2 = self.cat_embedding_1(cat)
        cat_3 = self.cat_embedding_2(cat)
        cat_4 = self.cat_embedding_3(cat)

        up0 = self.up0(latent_vec)
        up1 = self.up1(cat_1 * up0 + temb_1, down4)
        up2 = self.up2(cat_2 * up1 + temb_2, down3)
        up3 = self.up3(cat_3 * up2 + temb_3, down2)
        up4 = self.up4(cat_4 * up3 + temb_4, down1)

        return self._up(torch.cat((up4, x), 1))  # New

    def training_step(self, train_batch, batch_idx):
        self.last_batch = train_batch

        x = train_batch["image"].to(self.device)
        sentence = train_batch["encoded_sentence"].to(self.device)
        t = torch.randint(0, self.config["T"], (x.shape[0], 1), device=self.device)

        imgs_noisy, noise = self.image_diffuser(x, t)
        preds = self.forward(imgs_noisy, t, sentence)
        loss = self.compute_loss(noise, preds)

        self.log("train_loss", loss.detach().cpu().item())
        # print(f"Recent loss: {loss.detach().cpu().item():.5f}", end="\r")
        if batch_idx % 50 == 0:
            filename = self.plot_sample(self.last_batch)
            mlflow.log_artifact(filename)
        return loss

    def on_train_epoch_end(self):
        filename = self.plot_sample(self.last_batch)
        mlflow.log_artifact(filename)
        mlflow.pytorch.log_model(self, f"{int(time.time())}.pt")

    @torch.no_grad()
    def plot_sample(self, batch):
        # Take first image of batch
        sen = batch["encoded_sentence"][[0],]
        imgs = batch["image"][[0], :, :, :]

        img, _ = self.image_diffuser(imgs[[0], :, :, :], self.config["T"] - 1)
        org = torch.clone(img)
        for i in range(self.config["T"] - 1, -1, -1):
            _t = torch.tensor([[i]], device=img.device)
            img = self.image_diffuser.reverse(img, self.forward(img, _t, sen), _t)

            if i == (self.config["T"] - 1) // 2:
                halfway = torch.clone(img)

        nrows = 2
        ncols = 2
        samples = {
            "Original": imgs,
            "At the beginning": org,
            "Halfway": halfway,
            f"Predicted Original {np.argmax(sen.detach().cpu())}": img,
        }
        for i, (title, img) in enumerate(samples.items()):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.set_title(title)
            temp_vis(img)

        filename = f"/tmp/{int(time.time())}.png"
        plt.savefig(filename)
        return filename

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # def validation_step(self, val_batch, batch_idx):
    #     return
    #     x, y = val_batch["hsv"], val_batch["labels"].flatten().long()

    #     logits = self.forward(x)
    #     loss = self.compute_loss(logits, y)
    #     self.eval_loss.append(loss)

    #     self.eval_preds.append(logits.argmax(dim=-1))
    #     self.eval_true.append(y)

    # def on_validation_epoch_end(self):
    #     return
    #     avg_loss = torch.stack(self.eval_loss).cpu().mean()
    #     self.log("val_loss", avg_loss, sync_dist=True)
    #     self.eval_loss.clear()

    #     f1_score = -classification_report(
    #         torch.cat(self.eval_true).cpu(),
    #         torch.cat(self.eval_preds).cpu(),
    #         output_dict=True
    #     )["weighted avg"]["f1-score"]
    #     self.log("val_neg_f1", f1_score, sync_dist=True)
    #     self.eval_preds.clear()
    #     self.eval_true.clear()

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     print("on_before_opt enter")
    #     for name,p in self.named_parameters():
    #         if p.grad is None:
    #             print(name, p.shape)

    #     print("on_before_opt exit")
