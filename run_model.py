import numpy as np
import pandas as pd
import transformers
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
import random
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

torch.cuda.set_device(-1)
dist.init_process_group(backend='nccl')

print("Let's use", torch.cuda.device_count(), "GPUs!")


clusters = np.linspace(-5, 5, num=99)

data_clustered = np.load('data_clustered.npy')

class SongLyrics(Dataset):  
    def __init__(self, context):
        self.context = context
        
    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        x = torch.tensor(self.context[item]).long()
        return x
    
dataset = SongLyrics(data_clustered)  

vocab_size = len(clusters)
max_size = 512
# bos_token_id=vocab_size-4
# eos_token_id=vocab_size-3
# pad_token_id=vocab_size-2
# unk_token_id=vocab_size-1
    
config = transformers.GPT2Config(
    vocab_size=vocab_size,
#     bos_token_id=bos_token_id,
#     eos_token_id=eos_token_id,
#     pad_token_id=pad_token_id,
#     unk_token_id=unk_token_id,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
    unk_token_id=None,
    n_positions=max_size,
    n_ctx=max_size,
    n_embd=12*64,
    num_labels=vocab_size,
#     resid_pdrop=0,
#     embd_pdrop=0,
#     attn_pdrop=0,
)
model = transformers.GPT2LMHeadModel(config).cuda()
model = nn.DataParallel(model)
model.train()

epochs=100
warmup_steps=2000//(512/64)
# warmup_steps = 100
lr=0.000025
output_dir="./runs"
output_prefix="test"
save_model_on_epoch=False
time_steps = 30
    
device=torch.device("cuda")

batch_size_loader=8
batch_size = 512//batch_size_loader
train_dataloader = DataLoader(dataset, batch_size=batch_size_loader, shuffle=True)
loss=0
accumulating_batch_count = 0
input_tensor = None


optimizer = AdamW(model.parameters(), lr=lr)
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs*len(data_clustered)//512
)


for epoch in range(epochs):

    print(f"Training epoch {epoch}")
        
    pbar = tqdm(enumerate(train_dataloader), total=train_dataloader.__len__())
    mae_profit = [0]
    acc = [0]
    mae_cash = [0]
    for idx, input_tensor in pbar:
        input_tensor = input_tensor.to(device)
        
        if (accumulating_batch_count % batch_size) == 0:
            
            output = model.generate(
                input_ids=input_tensor[:, :-time_steps],
                max_length=max_size,temperature=1.0,do_sample=True,top_k=40, seed=42
            )
            sample = output[:,-time_steps:].cpu().detach().numpy()
            inp = input_tensor[:,-time_steps:].cpu().detach().numpy()
            acc.append((sample == inp).sum()/sample.size)

            profit_pred = clusters[sample]
            profit_true = clusters[inp]

            profit_pred = np.sign(profit_pred)*(np.exp(abs(profit_pred)) - 1)
            profit_true = np.sign(profit_true)*(np.exp(abs(profit_true)) - 1)
            
            mae_profit.append(mean_absolute_error(profit_true, profit_pred))

            cash_pred = 1.0
            cash_true = 1.0
            for p_pred, p_true in zip(profit_pred, profit_true):
                cash_pred *= 1+p_pred/100
                cash_true *= 1+p_true/100

            mae_cash.append(mean_absolute_error(cash_true-1, cash_pred-1))
        
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs[0]
        loss.backward()

        if (accumulating_batch_count % batch_size) == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            
        accumulating_batch_count += 1
        
        input_tensor = None
        pbar.set_postfix(loss=loss.cpu().detach().numpy(), ACC= acc[-1], mae_profit=mae_profit[-1], mae_cash=mae_cash[-1], 
                         lr=scheduler.get_last_lr()[0])
        
    if save_model_on_epoch:
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
        )
        
    print(f"Loss: {loss}, ACC: {np.mean(acc)}, MAE: {np.mean(mae_profit)}, MAE: {np.mean(mae_cash)}")
    print()