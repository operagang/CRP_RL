from datetime import datetime
import time
import os, sys
import argparse
import torch
import torch.optim as optim
import pandas as pd

try:
    from model.model import Model
    from generator.generator import Generator
    from trainer import train, eval, save_log, set_log, initialize, load_eval_data
    from benchmarks.benchmarks import solve_benchmarks, find_and_process_file
except:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from model.model import Model
    from generator.generator import Generator
    from trainer import train, eval, save_log, set_log, initialize, load_eval_data
    from benchmarks.benchmarks import solve_benchmarks, find_and_process_file





args = argparse.Namespace(
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),   # training device
    log_path = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",        # log / result directory
    load_model_path = './baselines/models/proposed/epoch(100).pt',             # pretrained model path
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




if __name__ == "__main__":
    model = Model(args).to(args.device)
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=args.device))
        print(f'* Model loaded: ({args.load_model_path})')


    model.eval()
    model.decoder.set_sampler('greedy')

    bays = [1,2,4,6,8,10]
    rows = [16]
    tiers = [6,8]

    data_path = './benchmarks/Lee_instances'

    instance_types = ['random', 'upsidedown']


    all_names = []
    all_wts = []

    for inst_type in instance_types:
        if inst_type == 'random':
            idxs = range(1,6)
        else:
            idxs = range(1,3)

        data_names = []
        wts = {}

        for tier in tiers:
            for row in rows:
                for bay in bays:
                    if tier == 8 and bay in [8, 10]:
                        continue

                    inputs, names = zip(*[
                        find_and_process_file(data_path, inst_type, bay, row, tier, idx, no_print=True)
                        for idx in idxs
                    ])
                    inputs = torch.cat(inputs)

                    s = time.time()
                    with torch.no_grad():
                        wt, _ = model(inputs.to(args.device), None)

                    all_names.extend([name.replace('.txt', '') for name in names])
                    all_wts.extend(wt.cpu().numpy())

                    print(names[0][:-8], wt.mean().item(), round((time.time()-s)/len(idxs),3))

    df = pd.DataFrame({
        'Instance': all_names,
        'WT': all_wts
    })
    df.to_excel("tmp_rl.xlsx", index=False)
