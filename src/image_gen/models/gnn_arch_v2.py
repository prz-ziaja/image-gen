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
import torch_geometric as tg
import torch_geometric.nn as tgnn
import torch.nn.functional as F
import mlflow
import image_gen.models.utils.gnn_utils as gu


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

        #####################################################
        self.num_layers = 4
        hidden_channels = 16
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            conv = tg.nn.HeteroConv(
                {
                    ('x', "to", 'x'): tg.nn.GATv2Conv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ('t', "to", 'x'): tg.nn.GATv2Conv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ('cat', "to", 'x'): tg.nn.GATv2Conv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
        
        self.out_linear = nn.Linear(hidden_channels,4)



        print(f"Device: {self.device}")
        self.lr = config["lr"]
        self.loss_function = config["loss_function"]
        self.cat_drop_prob = 0.1
        self._run()

    @torch.no_grad()
    def _run(self):
        image = torch.from_numpy(np.random.random([2,1,32,32])*128).to(self.device)
        category = torch.zeros(2,10).to(self.device)
        category[:,1] = 1
        timestep = torch.zeros(2,1).to(self.device)
        timestep[:,0] = 25
        _ = self.forward(x=image,t=timestep,cat=category)

    def compute_loss(self, imgs, imgs_pred):
        return self.loss_function(imgs, imgs_pred)

    def forward(self, x, t: int, cat: int):
        data = gu.build_gnn_batch(x=x, t=t, cat=cat)
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        for idx, conv in enumerate(self.convs):
            x_dict['x'] = conv(x_dict, {k:v[0].to(self.device) for k,v in edge_index_dict.items()})['x']
            #x_dict['x'] = conv(x_dict, edge_index_dict)['x']
            x_dict['x'] = F.gelu(x_dict['x'])

        output = self.out_linear(x_dict['x'])

        return gu.extract_prediction(output)

    def training_step(self, train_batch, batch_idx):
        self.last_batch = train_batch

        x = train_batch["image"].to(self.device)
        sentence = train_batch["encoded_sentence"].to(self.device)
        t = torch.randint(0, self.config["T"], (x.shape[0], 1), device=self.device)

        imgs_noisy, noise = self.image_diffuser(x, t)
        preds = self.forward(imgs_noisy, t, sentence)

        loss = self.compute_loss(noise, preds)

        self.log("train_loss", loss.detach().cpu().item())
        #print(f"train_loss {loss.detach().cpu().item()}")
        # print(f"Recent loss: {loss.detach().cpu().item():.5f}", end="\r")
        if batch_idx % 200 == 0:
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
