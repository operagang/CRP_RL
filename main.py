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
    lr = 0.000001,
    epochs = 500,

    batch_num = 100,
    batch_size = 64,
    mini_batch_num = 1,

    baseline = 'pomo', # or None
    pomo_size = 16,

    eval_path = './generator/eval_data/eval_data(35,2,4,6).pt',
    eval_batch_size = 1024,

    empty_priority = None, # or any integer
    norm_priority = True,
    add_fill_ratio = True,
    norm_layout = True,
    add_layout_ratio = True,
    add_travel_time = True,

    n_containers = 35,
    n_bays = 2,
    n_rows = 4,
    n_tiers = 6,
    instance_type = 'random',
    objective = 'workingtime', # or relocations

    load_model_path = './results/(~93)35,2,4,6,Norm,fill,N2/models/epoch(90).pt',

    embed_dim = 128,
    n_encode_layers = 3,
    n_heads = 8,
    ff_hidden = 512,
    tanh_c = 10,

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    log_path = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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

        train_loss = train(model, optimizer, args)
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
