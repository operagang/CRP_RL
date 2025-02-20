from datetime import datetime
import time
import argparse
import torch
import torch.optim as optim
from model import Model
from generator import Generator
from trainer import train, eval, save_log
from benchmarks import solve_benchmarks



args = argparse.Namespace(
    lr = 0.000001,
    epochs = 1500,
    batch_size = 512*2, # 256
    batch_num = 100, # 20
    eval_batch_size = 512*2, # 256
    eval_batch_num = 1, # 5
    eval_seed = 0,
    embed_dim = 128,
    n_encode_layers = 3,
    n_heads = 8,
    ff_hidden = 512,
    tanh_c = 10,
    n_bays = 2,
    n_rows = 4,
    n_tiers = 6,
    instance_type = 'random',
    objective = 'workingtime', # or relocations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    log_path = f"./train/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)




def main():
    model = Model(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_data = Generator(args, eval=True) # n_samples = eval_batch_size * eval_batch_num
    eval_data.data = torch.load('./eval_data.pt')
    clock = time.time()
    
    eval_wt, eval_reloc = eval(model, args, eval_data)
    clock = save_log(args, -1, None, eval_wt, eval_reloc, model, clock)
    solve_benchmarks(model, -1, args, ['random'])
    torch.save(eval_data.data, args.log_path + '/eval_data.pt')

    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, args)
        eval_wt, eval_reloc = eval(model, args, eval_data)
        clock = save_log(args, epoch, train_loss, eval_wt, eval_reloc, model, clock)
        
        if (epoch + 1) % 10 == 0:
            solve_benchmarks(model, epoch, args, ['random'])


if __name__ == "__main__":
    
    lrs = [
        # 1e-05
        5e-06,
        # 1e-06,
        # 5e-07,
        # 1e-07,
        # 5e-08
    ]
    for lr in lrs:
        args.lr = lr
        args.log_path = f"./train/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        main()
        pass
