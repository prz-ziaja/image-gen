import lightning.pytorch as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import image_gen.models.utils.blocks as blocks
import image_gen.models.utils.embeddings as embeddings
import image_gen.models.utils.image_diffusers as image_diffusers
from einops.layers.torch import Rearrange


import mlflow


def temp_vis(image):
    plt.imshow(image[0].permute([1, 2, 0]).detach().cpu())


class generator(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.noise_emb = self.latent_embed = embeddings.embedBlock(
            input_size=config["noise_vector_len"], layer_sizes=[256, 128]
        )
        IMG_SIZE = config["image_size"][0]
        IMG_CH = config["image_ch"]

        if IMG_SIZE == 32:
            self.img_upscale = nn.Sequential(
                blocks.upBlock2d(2, 8, 2),
                blocks.geluConv2d(8, 16),
                blocks.upBlock2d(16, 32, 2),
                nn.Conv2d(32, IMG_CH, (1, 1)),
            )
        elif IMG_SIZE == 64:
            self.img_upscale = nn.Sequential(
                blocks.upBlock2d(2, 32, 2),
                blocks.geluConv2d(32, 64),
                blocks.upBlock2d(64, 128, 2),
                blocks.geluConv2d(128, 256),
                blocks.upBlock2d(256, 64, 2),
                nn.Conv2d(64, IMG_CH, (1, 1)),
            )
        self.rearrange = Rearrange("b (ch h w) 1 1 -> b ch h w", ch=2, h=8, w=8)

    def forward(self, latents):
        x = self.noise_emb(latents)
        x = self.rearrange(x)

        x = self.img_upscale(x)

        return x


class discriminator(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        IMG_SIZE = config["image_size"][0]
        IMG_CH = config["image_ch"]

        self.img_downscale = nn.Sequential(
            blocks.downBlock2d(IMG_CH, 32, 1),
            blocks.geluConv2d(32, 64),
            blocks.downBlock2d(64, 64, 2),
            blocks.geluConv2d(64, 128),
            blocks.downBlock2d(128, 128, 2),
            blocks.geluConv2d(128, 256),
            blocks.downBlock2d(256, 256, 2),
            blocks.geluConv2d(256, 256),
            blocks.downBlock2d(256, 128, 2),
        )

        if IMG_SIZE == 32:
            linear_in = 1152
        elif IMG_SIZE == 64:
            linear_in = 2048

        self.linear = nn.Sequential(
            nn.Linear(linear_in, 256),
            nn.GELU(),
            nn.Linear(256, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        downscaled = self.img_downscale(x)
        downscaled_flat = downscaled.flatten(start_dim=1)
        classified = self.linear(downscaled_flat)
        return classified


class Model(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)

        self.generator = generator(config=config)
        self.critic = discriminator(config=config)
        self.config = config
        self.image_diffuser = image_diffusers.imageDiffuser(
            config["T"], config["t_start"], config["t_end"]
        )

        print(f"Device: {self.device}")
        self.noise_vector_len = 128  # config{"noise_vector_len"}
        self.lr = config["lr"]
        self.automatic_optimization = False

    def compute_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred)

    def forward(self, z):
        return self.generator(z)

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     print("on_before_opt enter")
    #     for name,p in self.named_parameters():
    #         if p.grad is None:
    #             print(name, p.shape)

    #     print("on_before_opt exit")
    def training_step(self, train_batch, batch_idx):
        optimizer_g, optimizer_c = self.optimizers()

        self.toggle_optimizer(optimizer_c)

        x = train_batch["image"].to(self.device)
        t = (
            torch.ones([x.shape[0], 1], dtype=torch.long, device=self.device)
            * self.config["image_noise_t"]
        )

        imgs_noisy, _ = self.image_diffuser(x, t)
        self.last_batch = torch.clone(imgs_noisy)

        z = torch.randn(imgs_noisy.shape[0], self.noise_vector_len, device=self.device)
        with torch.no_grad():
            fake = self(z)

        y_fake = torch.ones([fake.shape[0], 1], device=self.device)
        y_true = (-1) * torch.ones([imgs_noisy.shape[0], 1], device=self.device)
        y_critic = torch.cat([y_fake, y_true])
        x_critic = torch.cat([fake, imgs_noisy])

        critic_preds = self.critic(x_critic)
        critic_loss = self.compute_loss(y_critic, critic_preds)

        self.manual_backward(critic_loss)
        optimizer_c.step()
        optimizer_c.zero_grad()
        self.untoggle_optimizer(optimizer_c)

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        z = torch.randn(
            self.config["generated_image_per_step"],
            self.noise_vector_len,
            device=self.device,
        )
        y_generator = (-1) * torch.ones(
            [self.config["generated_image_per_step"], 1], device=self.device
        )

        fake = self(z)
        generator_preds = self.critic(fake)
        generator_loss = self.compute_loss(y_generator, generator_preds)

        self.manual_backward(generator_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        _g_loss = generator_loss.detach().cpu().item()
        _c_loss = critic_loss.detach().cpu().item()
        self.log("train_loss", (_c_loss + _g_loss) / 2)
        self.log("critic_loss", _c_loss)
        self.log("generator_loss", _g_loss)

    def on_train_epoch_end(self):
        filename = self.plot_sample(self.last_batch)
        mlflow.log_artifact(filename)
        mlflow.pytorch.log_model(self, f"{int(time.time())}.pt")

    @torch.no_grad()
    def plot_sample(self, imgs):
        # Take first image of batch
        z = torch.randn(3, self.noise_vector_len, device=self.device)
        fake = self(z)
        imgs = imgs[[0], :, :, :]

        nrows = 2
        ncols = 2
        samples = {
            "Original": imgs,
            "At the beginning": fake[[0]],
            "Halfway": fake[[1]],
            "Predicted Original": fake[[2]],
        }
        for i, (title, img) in enumerate(samples.items()):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.set_title(title)
            temp_vis(img)

        filename = f"/tmp/{int(time.time())}.png"
        plt.savefig(filename)
        return filename

    # def validation_step(self, val_batch, batch_idx):
    #     return
    #     x, y = val_batch["hsv"], val_batch["labels"].flatten().long()

    #     logits = self.forward(x)
    #     loss = self.compute_loss(logits, y)
    #     self.eval_loss.append(loss)

    #     self.eval_preds.append(logits.argmax(dim=-1))
    #     self.eval_true.append(y)

    # def on_validation_epoch_end(self):
    #     #mlflow.pytorch.log_model(self, f"{int(time.time())}.pt")
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

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        return [opt_g, opt_c], []
