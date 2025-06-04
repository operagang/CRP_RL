from datetime import datetime
import time
import os
import argparse
import torch
import torch.optim as optim
from model.model import Model
from generator.generator import Generator
from trainer import train, eval, save_log, set_log, initialize, load_eval_data
from benchmarks.benchmarks import solve_benchmarks



args = argparse.Namespace(

    epochs = 2000,

    bay_embedding = True, # bay embedding (avg. pooling) concat 할건지 말건지
    lstm = True, # LSTM 쓸건지 hand-crafted feature 쓸건지
    baseline = 'pomoZ', # \in {None, 'pomo', 'pomoZ'}

    train_data_idx = None, # multi-task learning -> None, 특정 layout -> Int
    train_data_sampler = 'uniform', # multi-task learning -> uniform, 특정 layout -> None

    # n_containers / (n_bays * n_rows * n_tiers) = 0.7~0.8 정도가 적당한 듯
    min_n_containers = [35,35,70,46], # 최소 컨테이너 수
    max_n_containers = [35,35,70,46], # 최대 컨테이너 수
    n_bays = [1,2,4,2], # bay 수
    n_rows = [8,4,4,4], # row 수
    n_tiers = [6,6,6,8], # stack 높이 (6 이랑 8 로 섞어)

    batch_size = 128,
    n_layouts_per_batch = 4,
    mini_batch_num = [2,2,2,2], # batch size 몇개로 잘라서 넣을건지



    #### 이 아래는 안건드려도 될 듯 ####
    lr = None,

    batch_num = 100,
    pomo_size = 16,

    eval_path = './generator/eval_data/eval_data(35,2,4,6).pt',
    eval_batch_size = 1024,

    empty_priority = None, # None or any integer (lstm 시 영향 X)
    norm_priority = True, # (lstm 시 영향 X)
    add_fill_ratio = True,
    norm_layout = True,
    add_layout_ratio = True,
    add_travel_time = True,

    instance_type = 'random',
    objective = 'workingtime', # workingtime or relocations

    load_model_path = None,

    embed_dim = 128,
    n_encode_layers = 3,
    n_heads = 8,
    ff_hidden = 512,
    tanh_c = 10,

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    log_path = None,
)




def main():
    model, optimizer, clock = initialize(args)
    eval_data = load_eval_data(args)

    ### test random model ###
    eval_wt, eval_reloc = eval(model, args, eval_data)
    clock = save_log(args, -1, None, eval_wt, eval_reloc, model, clock)
    solve_benchmarks(model, -1, args, ['random'])

    ### main loop ###
    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, args, epoch)
        eval_wt, eval_reloc = eval(model, args, eval_data)
        clock = save_log(args, epoch, train_loss, eval_wt, eval_reloc, model, clock)

        if (epoch + 1) % 10 == 0:
            solve_benchmarks(model, epoch, args, ['random'])


if __name__ == "__main__":
    
    lrs = [
        # 1e-05,
        5e-06,
        # 1e-06,
        # 5e-07,
        # 1e-07,
        # 5e-08
    ]
    for lr in lrs:
        args.lr = lr
        args.log_path = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        main()
        pass
