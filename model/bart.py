"""PyTorch Lighting bart model. """

import nlp2
import torch
import torch.utils.checkpoint
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.utils import logging
import pytorch_lightning as pl
import torch.utils.data as data

logger = logging.get_logger(__name__)


class Dataset(data.Dataset):
    def __init__(self, pool, min_len=1, max_len=1024):
        self.input_pool = [text for text in pool if min_len <= len(nlp2.split_sentence_to_array(text)) <= max_len]

    def __getitem__(self, index):
        return self.input_pool[index]

    def __len__(self):
        return len(self.input_pool)


# Denoising Pre-Training
class BARTDPTModel(pl.LightningModule):

    def __init__(self, config, train_datalist=None, eval_datalist=None, collate_fn=None, lr=3e-4, batch_size=10):
        super().__init__()
        # https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration
        self.bart = BartForConditionalGeneration.from_pretrained(config)
        self.tokenizer = BartTokenizer.from_pretrained(config)
        self.config = config
        self.lr = lr
        self.train_datalist = train_datalist
        self.eval_datalist = eval_datalist
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def train_dataloader(self):
        loader = data.DataLoader(Dataset(self.train_datalist), batch_size=self.batch_size, shuffle=True,
                                 collate_fn=self.collate_fn)
        return loader

    def val_dataloader(self):
        loader = data.DataLoader(Dataset(self.eval_datalist), batch_size=self.batch_size, shuffle=False,
                                 collate_fn=self.collate_fn)
        return loader

    def forward(self, input_text):
        outputs = self.bart(
            **self.tokenizer(input_text, return_tensors='pt').to(self.device)
        )
        return outputs

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        loss = self.bart(
            batch.input_ids,
            labels=batch.labels
        )[0]
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        # self.log('dev_loss', loss, prog_bar=True)
        self.log('dev_loss', loss, prog_bar=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=(self.lr or self.learning_rate))
