import os
from datetime import datetime
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Model
from generator import Generator


def save_log(args, epoch, loss, wt, reloc, model, clock):
    if epoch == -1:
        os.makedirs(args.log_path)
        os.makedirs(args.log_path + '/models')
        with open(args.log_path + '/log.txt', 'w') as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write('--------------------\n')
    new_clock = time.time()
    message = f'Epoch: {epoch+1} | Train loss: {loss} | Eval WT: {round(wt, 3)} | Eval moves: '\
            + f'{round(reloc + args.n_containers, 3)} '\
            + f'| {round(new_clock-clock)}s'
    with open(args.log_path + '/log.txt', 'a') as f:
        f.write(message + '\n')
    torch.save(model.state_dict(), args.log_path + f'/models/epoch({epoch + 1}).pt')
    print(message)
    return new_clock


def get_loss(args, wt, ll, reloc):
    if args.objective == 'workingtime':
        obj = wt
    elif args.objective == 'relocations':
        obj = reloc
    else:
        raise ValueError('obj 오류')
    
    if args.baseline is None:
        return (obj * ll).mean()
    elif args.baseline == 'pomo':
        obj_reshaped = obj.view(args.batch_size, args.pomo_size)
        obj_mean = obj_reshaped.mean(dim=1, keepdim=True)
        obj_adjusted = (obj_reshaped - obj_mean).view(obj.shape[0])
        return (obj_adjusted * ll).mean()
    # else:
    #     obj_mean = obj.mean()
    #     obj_std = obj.std(unbiased=False)  # 작은 배치에서도 안정적이도록 `unbiased=False` 사용
    #     norm_obj = (obj - obj_mean) / (obj_std + 1e-8)  # 0으로 나누는 문제 방지
    #     return (norm_obj * ll).mean()


def train(model, optimizer, args):
    model.train()
    if args.baseline == 'pomo':
        model.decoder.set_sampler('sampling')
    else:
        model.decoder.set_sampler('greedy')

    dataset = Generator(args) # n_samples = batch_size * batch_num
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    tbar = tqdm(dataloader)
    losses = []

    for batch in tbar:

        if args.baseline == 'pomo':
            batch = batch.unsqueeze(1).expand(batch.shape[0], args.pomo_size, batch.shape[1], batch.shape[2]).reshape(batch.shape[0] * args.pomo_size, batch.shape[1], batch.shape[2])
            wt, ll, reloc = model(batch.to(args.device), dataset.n_bays, dataset.n_rows)
        else:
            wt, ll, reloc = model(batch.to(args.device), dataset.n_bays, dataset.n_rows)

        loss = get_loss(args, wt, ll, reloc)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        losses.append(loss.item())
        tbar.set_description('Train loss: %.5f' % (np.mean(losses)))

    return np.mean(losses)


def eval(model, args, dataset):
    clock = time.time()
    torch.cuda.empty_cache()
    model.eval()
    model.decoder.set_sampler('greedy')

    eval_loader = DataLoader(dataset=dataset, batch_size=args.eval_batch_size)

    wts = []; relocs = []
    for batch in eval_loader:
        with torch.no_grad():
            wt, _, reloc = model(batch.to(args.device), dataset.n_bays, dataset.n_rows)
            wts.extend(wt.tolist())
            relocs.extend(reloc.tolist())
    print(f'Eval 시간: {round(time.time() - clock, 1)}s')
    return np.mean(wts), np.mean(relocs)




if __name__ == '__main__':

    pass


