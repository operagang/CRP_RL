import os
from datetime import datetime
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import Model
from generator.generator import Generator


def check_args_validity(args):
    assert len(args.mini_batch_num) == len(args.min_n_containers) == len(args.max_n_containers)\
    == len(args.n_bays) == len(args.n_rows) == len(args.n_tiers)
    if args.train_data_sampler is not None:
        assert args.train_data_idx is None
    if args.train_data_idx is not None:
        assert args.train_data_sampler is None
    if args.baseline in ['pomo', 'pomoZ']:
        assert args.pomo_size > 0
    if args.pomo_size is not None:
        assert args.baseline in ['pomo', 'pomoZ']


def initialize(args):
    check_args_validity(args)
    print(f'* Device: {args.device}')
    model = Model(args).to(args.device)
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path))
        print(f'* Model loaded: ({args.load_model_path})')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    set_log(args)
    clock = time.time()
    print(f'* lr: {args.lr}, epochs: {args.epochs}')
    print(f'* batch_num: {args.batch_num}, batch_size: {args.batch_size}, mini_batch_num: {args.mini_batch_num}')
    print(f'* baseline: {args.baseline}, pomo_size: {args.pomo_size}')
    print(f'* empty_priority: {args.empty_priority}, norm_priority: {args.norm_priority}')
    return model, optimizer, clock


def load_eval_data(args):
    eval_data = Generator(load_data=args.eval_path)
    torch.save(eval_data.data, args.log_path + '/eval_data.pt')
    print(f'* eval data size = {eval_data.data.shape}')
    return eval_data


def set_log(args):
    os.makedirs(args.log_path)
    os.makedirs(args.log_path + '/models')
    with open(args.log_path + '/log.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write('--------------------\n')


def save_log(args, epoch, loss, wt, reloc, model, clock):
    new_clock = time.time()
    message = f'Epoch: {epoch+1} | Train loss: {loss} | Eval WT: {round(wt, 3)} | Eval relocs: '\
            + f'{round(reloc, 3)} '\
            + f'| {round(new_clock-clock)}s'
    with open(args.log_path + '/log.txt', 'a') as f:
        f.write(message + '\n')
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), args.log_path + f'/models/epoch({epoch + 1}).pt')
    print(message)
    return new_clock


def get_loss(args, wt, ll, reloc, idx):
    if args.objective == 'workingtime':
        obj = wt
    elif args.objective == 'relocations':
        obj = reloc
    else:
        raise ValueError('obj 오류')
    
    if args.baseline is None:
        return (obj * ll).mean()
    elif args.baseline == 'pomo':
        obj_reshaped = obj.view(args.batch_size // args.mini_batch_num[idx], args.pomo_size)
        obj_mean = obj_reshaped.mean(dim=1, keepdim=True)
        obj_adjusted = (obj_reshaped - obj_mean).view(obj.shape[0])
        return (obj_adjusted * ll).mean()
    elif args.baseline == 'pomoZ':
        obj_reshaped = obj.view(args.batch_size // args.mini_batch_num[idx], args.pomo_size)
        obj_mean = obj_reshaped.mean(dim=1, keepdim=True)
        obj_std = obj_reshaped.std(dim=1, keepdim=True, unbiased=False)  # 작은 배치에서도 안정적이도록 `unbiased=False` 사용
        obj_adjusted = ((obj_reshaped - obj_mean) / (obj_std + 1e-8)).view(obj.shape[0])
        return (obj_adjusted * ll).mean()
    # else:
    #     obj_mean = obj.mean()
    #     obj_std = obj.std(unbiased=False)  # 작은 배치에서도 안정적이도록 `unbiased=False` 사용
    #     norm_obj = (obj - obj_mean) / (obj_std + 1e-8)  # 0으로 나누는 문제 방지
    #     return (norm_obj * ll).mean()


def sample_data_idx(args):
    if args.train_data_idx is not None:
        return args.train_data_idx

    if args.train_data_sampler == 'uniform':
        return random.randint(0, len(args.max_n_containers) - 1)


def train(model, optimizer, args):
    model.train()
    if args.baseline in ['pomo', 'pomoZ']:
        model.decoder.set_sampler('sampling')
    else:
        model.decoder.set_sampler('greedy')
    
    idx = sample_data_idx(args)
    n_containers = random.randint(args.min_n_containers[idx], args.max_n_containers[idx])
    layout = (n_containers, args.n_bays[idx], args.n_rows[idx], args.n_tiers[idx])
    print(f'Train data layout = {layout}')

    dataset = Generator(
        n_samples=args.batch_size * args.batch_num,
        layout=layout,
        inst_type=args.instance_type,
        device=args.device
    )
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    tbar = tqdm(dataloader)
    losses = []
    optimizer.zero_grad()

    for batch in tbar: # batch_num
        accumulated_loss = 0.0

        for i in range(args.mini_batch_num[idx]):
            assert args.batch_size % args.mini_batch_num[idx] == 0.0
            start_idx = i * (args.batch_size // args.mini_batch_num[idx])
            end_idx = (i + 1) * (args.batch_size // args.mini_batch_num[idx])
            mini = batch[start_idx:end_idx]  # (batch_size, ...)

            if args.baseline in ['pomo', 'pomoZ']:
                mini_expanded = mini.unsqueeze(1).expand(mini.shape[0], args.pomo_size, mini.shape[1], mini.shape[2], mini.shape[3])
                mini_expanded = mini_expanded.reshape(mini.shape[0] * args.pomo_size, mini.shape[1], mini.shape[2], mini.shape[3])
                wt, ll, reloc = model(mini_expanded.to(args.device))
            else:
                wt, ll, reloc = model(mini.to(args.device))

            loss = get_loss(args, wt, ll, reloc, idx) / args.mini_batch_num[idx]  # Gradient Accumulation을 위한 평균화
            loss.backward()  # `loss.backward()`는 여기서 실행
            accumulated_loss += loss.item()  # Loss 저장

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)  # Gradient Clipping
        optimizer.step()  # `mini_batch_num`번 누적한 후 한 번만 실행
        optimizer.zero_grad()  # Gradient 초기화

        losses.append(accumulated_loss)
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
            wt, _, reloc = model(batch.to(args.device))
            wts.extend(wt.tolist())
            relocs.extend(reloc.tolist())
    print(f'Eval 시간: {round(time.time() - clock, 1)}s')
    return np.mean(wts), np.mean(relocs)




if __name__ == '__main__':

    pass


