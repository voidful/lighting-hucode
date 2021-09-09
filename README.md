# lighting-hucode

## Usage

modify config in `train.py`

```python
model_config = "facebook/bart-base"
batch_size = 50
train_list = lista # list, sentences in line
eval_list = listb # list, sentences in line

# and
pl.Trainer() # training config
```

run

```shell
python train.py
```