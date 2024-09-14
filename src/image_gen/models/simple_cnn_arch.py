import lightning.pytorch as pl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


class Model(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        self.model = timm.create_model(**config)

        self.eval_loss = []
        self.eval_preds = []
        self.eval_true = []
        self.loss_function = config["loss_function"]#nn.CrossEntropyLoss()#HingeEmbeddingLoss()

    def compute_loss(self, logits, labels):
        return self.loss_function(logits, labels)

    def forward(self, x):
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x"""

        x = self.model(x)

        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["hsv"], train_batch["labels"].flatten().long()
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["hsv"], val_batch["labels"].flatten().long()

        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        self.eval_loss.append(loss)

        self.eval_preds.append(logits.argmax(dim=-1))
        self.eval_true.append(y)
        

    def on_validation_epoch_end(self):
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
