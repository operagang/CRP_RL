from datetime import datetime
import argparse
import torch
from trainer import train, save_log, initialize
from benchmarks.benchmarks import solve_benchmarks



args = argparse.Namespace(
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),   # training device
    log_path = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",        # log / result directory
    load_model_path = None,                                                    # pretrained model path
    epochs = 1000,                                                             # number of training epochs
    batch_num = 100,                                                           # number of batches per epoch
    batch_size = 128,                                                          # batch size
    pomo_size = 16,                                                            # POMO rollout size
    lr = 3*1e-4,                                                               # learning rate
    baseline = 'proposed',                                                     # baseline type: {None, 'pomo', 'proposed'}
    instance_type = 'random',                                                  # training instance type
    n_layouts_per_batch = 4,                                                   # number of layouts per batch
    min_n_containers = 35,                                                     # minimum number of containers in traning instances
    max_n_containers = 70,                                                     # maximum number of containers in traning instances
    embed_dim = 128,                                                           # embedding dimension
    n_encode_layers = 3,                                                       # number of encoder layers
    n_heads = 8,                                                               # number of attention heads
    ff_hidden = 512,                                                           # hidden dimension of feed-forward layers
    tanh_c = 10,                                                               # tanh clipping value
    lstm = True,                                                               # use LSTM (otherwise hand-crafted features)
    bay_embedding = True,                                                      # concatenate bay embedding
    online = False,                                                            # enable online setting
    online_known_num = None                                                    # number of known future requests (online)
)





def main():
    model, optimizer, clock = initialize(args)
    clock = save_log(args, -1, None, model, clock)
    solve_benchmarks(model, -1, args, instance_types=['random'])

    for epoch in range(args.epochs):

        train_loss = train(model, optimizer, args, epoch)
        clock = save_log(args, epoch, train_loss, model, clock)
        if (epoch + 1) % 1 == 0:
            solve_benchmarks(model, epoch, args, instance_types=['random'])


if __name__ == "__main__":
    main()
