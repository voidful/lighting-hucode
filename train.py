import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from transformers import (
    AutoTokenizer
)
from model.bart import BARTDPTModel

import pytorch_lightning as pl

model_config = "facebook/bart-base"
batch_size = 50
train_list = open('train.txt').read().splitlines()
eval_list = open('test.txt').read().splitlines()


def dataset_collate(batch):
    inputs = tokenizer(batch, return_tensors="pt",
                       padding=True)
    inputs['labels'] = inputs['input_ids']
    inputs_id = inputs['input_ids']
    mask = torch.ones(inputs_id.shape).uniform_() > 0.25
    result = inputs_id * mask
    result[result == 0] = tokenizer.mask_token_id
    eos = torch.tensor([tokenizer.eos_token_id]).repeat(result.shape[0]).unsqueeze(1)
    bos = torch.tensor([tokenizer.bos_token_id]).repeat(result.shape[0]).unsqueeze(1)
    mask = torch.tensor([-100]).repeat(result.shape[0]).unsqueeze(1)
    result = torch.cat((bos, result, eos), dim=1)
    inputs['labels'] = torch.cat((mask, inputs['labels'], mask), dim=1)
    inputs['input_ids'] = result
    inputs.pop('attention_mask')
    return inputs


tokenizer = AutoTokenizer.from_pretrained(model_config)
bart_dpt = BARTDPTModel(model_config, train_list, eval_list,
                        dataset_collate,
                        batch_size=batch_size)

es = EarlyStopping(monitor='dev_loss')
trainer = pl.Trainer(gpus=2, check_val_every_n_epoch=1, callbacks=[es, ModelCheckpoint(
    monitor='dev_loss', filename='{epoch}-{dev_loss:.2f}', save_last=True, )],
                     default_root_dir='./bart_dpt/', auto_lr_find=True, accelerator='ddp2',
                     plugins=DDPPlugin(find_unused_parameters=True))
trainer.tune(bart_dpt)
trainer.fit(bart_dpt)
