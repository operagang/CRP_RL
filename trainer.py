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
import gc
from model.model import Model
from generator.generator import Generator


def check_args_validity(args):
    if args.baseline in ['pomo', 'pomoZ']:
        assert args.pomo_size > 0
    if args.pomo_size is not None:
        assert args.baseline in ['pomo', 'pomoZ']


def initialize(args):
    #check_args_validity(args)
    print(f'* Device: {args.device}')
    model = Model(args).to(args.device)
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path))
        print(f'* Model loaded: ({args.load_model_path})')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    set_log(args)
    clock = time.time()
    print(f'* lr: {args.lr}, epochs: {args.epochs}')
    print(f'* batch_num: {args.batch_num}, batch_size: {args.batch_size}')
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
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), args.log_path + f'/models/epoch({epoch + 1}).pt')
    print(message)
    return new_clock


def get_loss(args, wt, ll, reloc, mini_batch_num):
    if args.objective == 'workingtime':
        obj = wt
    elif args.objective == 'relocations':
        obj = reloc
    else:
        raise ValueError('obj 오류')
    
    if args.baseline is None:
        return (obj * ll).mean()
    elif args.baseline == 'pomo':
        obj_reshaped = obj.view(args.batch_size // args.n_layouts_per_batch // mini_batch_num, args.pomo_size)
        obj_mean = obj_reshaped.mean(dim=1, keepdim=True)
        obj_adjusted = (obj_reshaped - obj_mean).view(obj.shape[0])
        return (obj_adjusted * ll).mean()
    elif args.baseline == 'pomoZ':
        obj_reshaped = obj.view(args.batch_size // args.n_layouts_per_batch // mini_batch_num, args.pomo_size)
        obj_mean = obj_reshaped.mean(dim=1, keepdim=True)
        obj_std = obj_reshaped.std(dim=1, keepdim=True, unbiased=False)  # 작은 배치에서도 안정적이도록 `unbiased=False` 사용
        obj_adjusted = ((obj_reshaped - obj_mean) / (obj_std + 1e-8)).view(obj.shape[0])
        return (obj_adjusted * ll).mean()
    # else:
    #     obj_mean = obj.mean()
    #     obj_std = obj.std(unbiased=False)  # 작은 배치에서도 안정적이도록 `unbiased=False` 사용
    #     norm_obj = (obj - obj_mean) / (obj_std + 1e-8)  # 0으로 나누는 문제 방지
    #     return (norm_obj * ll).mean()


# def sample_data_idx(args):
#     if args.train_data_idx is not None:
#         return args.train_data_idx

#     if args.train_data_sampler == 'uniform':
#         return random.randint(0, len(args.max_n_containers) - 1)

def sample_layout(min_n_containers, max_n_containers, utilization_range=(0.6, 0.8)):
    while True:
        n_containers = random.randint(min_n_containers, max_n_containers)
        
        n_tiers = random.choice([6, 8]) # tier는 6 or 8 고정
        min_total = int(n_containers / utilization_range[1])
        max_total = int(n_containers / utilization_range[0])

        possible_pairs = []
        for total_slots in range(min_total, max_total + 1):
            if total_slots % n_tiers != 0: # int check
                continue
            area = total_slots // n_tiers # bay x row
            for n_bays in range(1, area + 1): # 가능한 모든 쌍 찾기
                if area % n_bays == 0:
                    n_rows = area // n_bays
                    if n_rows > n_bays and n_rows <= 16:  # 조건 추가
                        possible_pairs.append((n_bays, n_rows))
                    

        if possible_pairs:
            n_bays, n_rows = random.choice(possible_pairs)
            return n_containers, n_bays, n_rows, n_tiers
        else:
            return 35, 2, 4, 6 # temp

def train(model, optimizer, args, epoch):
    model.train()
    if args.baseline in ['pomo', 'pomoZ']:
        model.decoder.set_sampler('sampling')
    else:
        model.decoder.set_sampler('greedy')

    losses = []
    optimizer.zero_grad()
    tbar = tqdm(range(args.batch_num), desc="Training")
    
    for step in tbar:
        accumulated_loss = 0.0

        for i in range(args.n_layouts_per_batch):
            if i >= args.large_n_layouts_per_batch:
                n_containers, n_bays, n_rows, n_tiers = sample_layout(min_n_containers=args.min_n_containers, max_n_containers=args.max_n_containers)
                if n_containers<37:
                    mini_batch_num = 1
                else:
                    mini_batch_num = 2
                max_retrievals = None
            else:
                n_containers, n_bays, n_rows, n_tiers = sample_layout(min_n_containers=args.large_min_n_containers, max_n_containers=args.large_max_n_containers)
                mini_batch_num = 4
                assert args.max_retrievals is not None
                max_retrievals = args.max_retrievals
            assert type(args.batch_size // args.n_layouts_per_batch // mini_batch_num) == int
            layout = (n_containers, n_bays, n_rows, n_tiers)
            
            for _ in range(mini_batch_num):
                mini = Generator(
                    n_samples=args.batch_size // args.n_layouts_per_batch // mini_batch_num, # int 인거 미리 검사됨
                    layout=layout,
                    inst_type=args.instance_type,
                    device=args.device
                )[:]

                if args.baseline in ['pomo', 'pomoZ']:
                    mini_expanded = mini.unsqueeze(1).expand(mini.shape[0], args.pomo_size, mini.shape[1], mini.shape[2], mini.shape[3])
                    mini_expanded = mini_expanded.reshape(mini.shape[0] * args.pomo_size, mini.shape[1], mini.shape[2], mini.shape[3])
                    wt, ll, reloc, wt_lb = model(mini_expanded.to(args.device), max_retrievals)
                else:
                    wt, ll, reloc, wt_lb = model(mini.to(args.device), max_retrievals)
                wt = (wt + args.lower_bound_weight * wt_lb).to(args.device)

                loss = get_loss(args, wt, ll, reloc, mini_batch_num) / args.n_layouts_per_batch / mini_batch_num
                loss.backward()  # `loss.backward()`는 여기서 실행
                accumulated_loss += loss.item()  # Loss 저장

            if epoch == 0 and step < 100:
                del loss
                gc.collect()
                torch.cuda.empty_cache()


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
            wt, _, reloc, _ = model(batch.to(args.device), None)
            wts.extend(wt.tolist())
            relocs.extend(reloc.tolist())
    print(f'Eval 시간: {round(time.time() - clock, 1)}s')
    return np.mean(wts), np.mean(relocs)




if __name__ == '__main__':

    pass


