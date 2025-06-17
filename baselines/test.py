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

    epochs = 1000,

    bay_embedding = True, # bay embedding (avg. pooling) concat 할건지 말건지
    lstm = True, # LSTM 쓸건지 hand-crafted feature 쓸건지
    baseline = 'pomoZ', # \in {None, 'pomo', 'pomoZ'}

    # train_data_idx = None, # multi-task learning -> None, 특정 layout -> Int
    # train_data_sampler = 'uniform', # multi-task learning -> uniform, 특정 layout -> None

    batch_size = 128,
    n_layouts_per_batch = 4,
    min_n_containers = 35, # 최소 컨테이너 수
    max_n_containers = 70, # 최대 컨테이너 수

    large_n_layouts_per_batch = 1,
    large_min_n_containers = 70, # 최소 컨테이너 수
    large_max_n_containers = 140, # 최대 컨테이너 수
    max_retrievals = 70,
    lower_bound_weight = 1,

    load_model_path = './baselines/models/proposed/epoch(100).pt',

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

    embed_dim = 128,
    n_encode_layers = 3,
    n_heads = 8,
    ff_hidden = 512,
    tanh_c = 10,

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    log_path = None,
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

    # bays = [20]
    # rows = [16]
    # tiers = [6,8]

    data_path = './benchmarks/Lee_instances'
    # data_path = './benchmarks/Shin_instances'

    instance_types = ['random', 'upsidedown']


    all_names = []
    all_wts = []


    # """"""
    # from collections import defaultdict
    # data_by_instance = defaultdict(dict)


    for inst_type in instance_types:
        idxs = range(1,6) if inst_type == 'random' else range(1,3)
        # idxs = range(1,21)

        data_names = []
        wts = {}
        moves = {}

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
                        wt, _, reloc, _ = model(inputs.to(args.device), None)

                    # 이름 정리 및 결과 저장
                    all_names.extend([name.replace('.txt', '') for name in names])
                    all_wts.extend(wt.cpu().numpy())  # GPU에서 CPU로 이동

                    print(names[0][:-8], wt.mean().item(), round((time.time()-s)/len(idxs),3))


                    # """"""
                    # data_by_instance[names[0][:-8]]['well_located'] = model.decoder.well_located
                    # data_by_instance[names[0][:-8]]['bay_diff'] = model.decoder.bay_diff
                    # data_by_instance[names[0][:-8]]['row_diff'] = model.decoder.row_diff
                    # model.decoder.well_located = []
                    # model.decoder.bay_diff = []
                    # model.decoder.row_diff = []

    # DataFrame 생성 및 Excel 저장
    df = pd.DataFrame({
        'Instance': all_names,
        'WT': all_wts
    })
    df.to_excel("tmp_rl.xlsx", index=False)




    # """"""
    # import numpy as np
    # summary_rows = []
    # all_bay_diffs = set()
    # all_row_diffs = set()

    # # First pass to collect all diff values
    # for instance, data in data_by_instance.items():
    #     all_bay_diffs.update(data["bay_diff"])
    #     all_row_diffs.update(data["row_diff"])

    # bay_diff_keys = sorted(all_bay_diffs)
    # row_diff_keys = sorted(all_row_diffs)

    # # Second pass to build summary
    # for instance, data in data_by_instance.items():
    #     row = {"Instance": instance}
    #     well_located = np.array(data["well_located"])
    #     bay_diff = np.array(data["bay_diff"])
    #     row_diff = np.array(data["row_diff"])

    #     row["Total"] = len(well_located)
    #     row["WellLocated"] = well_located.sum()

    #     for diff in bay_diff_keys:
    #         mask = bay_diff == diff
    #         row[f"BayDiff_{diff}_False"] = np.logical_and(mask, ~well_located).sum()
    #         row[f"BayDiff_{diff}_True"] = np.logical_and(mask, well_located).sum()

    #     for diff in row_diff_keys:
    #         mask = row_diff == diff
    #         row[f"RowDiff_{diff}_False"] = np.logical_and(mask, ~well_located).sum()
    #         row[f"RowDiff_{diff}_True"] = np.logical_and(mask, well_located).sum()

    #     summary_rows.append(row)

    # # Create DataFrame
    # df_summary = pd.DataFrame(summary_rows)
    # df_summary.to_excel("log_rl.xlsx", index=False)