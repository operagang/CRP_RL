import torch

def count_disorder_per_row(x):
    max_tiers = x.shape[-1]

    i = torch.arange(max_tiers, device=x.device).view(1, -1, 1)  # shape (1, S, 1)
    j = torch.arange(max_tiers, device=x.device).view(1, 1, -1)  # shape (1, 1, S)
    mask = j < i

    x_expanded = x.unsqueeze(2)  # shape (R, S, 1)
    x_below = x.unsqueeze(1)     # shape (R, 1, S)

    valid = (x_expanded != 0) & (x_below != 0) & mask

    compare = (x_expanded > x_below) & valid

    disorder_flag = compare.any(dim=2)  # shape (R, S)

    count = disorder_flag.sum(dim=1)
    return count


def get_wt_lb(x):
    n_bays = x.shape[1]
    n_rows = x.shape[2]
    t_pd = 30
    t_acc = 40
    t_bay = 3.5
    t_row = 1.2


    x = x.squeeze(0)  
    x = x.view(-1, x.size(-1))

    curr_bay = None
    curr_row = None
    
    nonzero_pos = (x != 0).nonzero(as_tuple=False)

    values = x[nonzero_pos[:, 0], nonzero_pos[:, 1]]
    stacks = nonzero_pos[:, 0]

    sorted_indices = values.argsort()
    sorted_stacks = stacks[sorted_indices]

    lb1 = 0.0

    for stack_tensor in sorted_stacks:
        stack = stack_tensor.item()
        next_bay = stack // n_rows + 1
        next_row = stack % n_rows + 1

        if curr_bay is None:
            curr_bay = next_bay
        if curr_row is None:
            curr_row = next_row

        if curr_bay != next_bay:
            lb1 += t_acc
            lb1 += t_bay * abs(curr_bay - next_bay)
        lb1 += t_row * abs(curr_row - next_row)
        lb1 += t_row * next_row
        lb1 += t_pd

        curr_bay = next_bay
        curr_row = 0

    lb2 = (2 * t_row + t_pd) * count_disorder_per_row(x).sum().item()

    return lb1 + lb2



if __name__ == "__main__":
    try:
        from benchmarks.benchmarks import find_and_process_file
    except:
        import os, sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if top_level_dir not in sys.path:
            sys.path.append(top_level_dir)
        from benchmarks.benchmarks import find_and_process_file

    folder_path = "./benchmarks/Lee_instances"  # Replace with the folder containing your files
    n_rows = 16
    results = []
    for inst_type in ['random', 'upsidedown']:
        for n_tiers in [6,8]:
            for n_bays in [1,2,4,6,8,10]:
                for id in range(1,6):
                    if n_tiers == 8 and n_bays in [8, 10]:
                        continue
                    if inst_type == 'upsidedown' and id in [3,4,5]:
                        continue

                    container_tensor, inst_name = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

                    cost = get_wt_lb(container_tensor) # batch X

                    print(f'inst_name: {inst_name}')
                    print(f'cost: {cost}')

                    results.append([inst_name, cost])
    
    import pandas as pd
    df = pd.DataFrame(results, columns=["inst_name", "WT"])
    df.to_excel('./tmp_LB.xlsx', index=False)