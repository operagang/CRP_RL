from datetime import datetime
import argparse
import torch
from trainer import train, eval, save_log, initialize
from benchmarks.benchmarks import solve_benchmarks



args = argparse.Namespace(
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    log_path = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    load_model_path = None,
    epochs = 1000,
    batch_num = 100,
    batch_size = 128,
    pomo_size = 16,
    lr = 3*1e-4,
    baseline = 'pomoZ', # \in {None, 'pomo', 'pomoZ'}
    instance_type = 'random',
    n_layouts_per_batch = 4, # batch_size를 몇개의 layout으로 나누어 학습 할 것인지
    min_n_containers = 35, # 최소 컨테이너 수
    max_n_containers = 70, # 최대 컨테이너 수
    embed_dim = 128,
    n_encode_layers = 3,
    n_heads = 8,
    ff_hidden = 512,
    tanh_c = 10,
    lstm = True, # LSTM 쓸건지 hand-crafted feature 쓸건지
    bay_embedding = True, # bay embedding (avg. pooling) concat 할건지 말건지
    online = False,
    online_known_num = None
)




def main():
    model, optimizer, clock = initialize(args)

    ### test random model ###
    clock = save_log(args, -1, None, model, clock)
    solve_benchmarks(model, -1, args, instance_types=['random']) # 문헌의 벤치마크 문제로 test

    ### main loop ###
    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, args, epoch)
        clock = save_log(args, epoch, train_loss, model, clock)

        if (epoch + 1) % 1 == 0:
            solve_benchmarks(model, epoch, args, instance_types=['random'])


if __name__ == "__main__":
    main()
