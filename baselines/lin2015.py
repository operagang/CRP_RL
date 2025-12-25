import torch
try:
    from env.env import Env
except:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from env.env import Env



class Lin2015():
    def __init__(self, pr=30, pb=300):
        self.pr = pr # SSI row weight
        self.pb = pb # SSI bay weight

    def get_top_priority(self, x):
        _, num_stacks, max_tiers = x.shape  # (batch, stack, tiers)
        top_idxs = (x > 0).sum(dim=2) - 1
        top_priorities = torch.where(
            top_idxs >= 0,
            x[torch.arange(1).view(-1, 1), torch.arange(num_stacks).view(1, -1), top_idxs], 
            torch.tensor(num_stacks * max_tiers * 10, device=x.device, dtype=x.dtype)
        )
        return top_priorities

    def get_target_stack(self, x):
        _, num_stacks, max_tiers = x.shape  # (batch, stack, tiers)
        mn_val = torch.min(torch.where(x == .0, torch.FloatTensor([1+num_stacks*max_tiers]), x), dim=2)[0]
        target_stack = torch.argmin(mn_val, dim=1)
        return target_stack

    def check_validity(self, x):
        # 1. find non-zero values
        nonzero_values = x[x > 0].int().tolist()
        n = len(nonzero_values)

        # 2. check exactly the values 1 to n exist
        required_values = set(range(1, n + 1))
        is_valid = set(nonzero_values) == required_values

        # 3. check each stack has "0s after the first non-zero value"
        for stack in x:
            nonzero_idxs = (stack > 0).nonzero(as_tuple=True)[0]
            if len(nonzero_idxs) > 0:
                last_nonzero_idx = nonzero_idxs[-1].item()
                if (stack[last_nonzero_idx + 1:] > 0).any():
                    is_valid = False
                    break

        if not is_valid:
            raise ValueError(f"not valid env.x: \n{x}")

        return

    def run(self, x, online=False, online_container_num=None, restricted=False):
        _, n_bays, n_rows, max_tiers = x.shape  # (1, bays, rows, tiers)
        n_containers = torch.sum(x > 0).item()
        device = torch.device('cpu')

        env = Env(device, x)
        cost = torch.zeros(env.x.shape[0])

        while True:
            cost += env.clear()

            self.check_validity(env.x.squeeze())

            if env.all_empty():
                break

            masked_x = env.x.clone()

            if online:
                mask = masked_x > online_container_num
                for b in range(masked_x.shape[0]):
                    masked_x[b][mask[b]] = online_container_num + 1

            stack_len = (masked_x > 0).sum(dim=2).squeeze()
            valid_stacks = stack_len < max_tiers

            top_priorities = self.get_top_priority(masked_x).squeeze()
            target_stack = self.get_target_stack(masked_x)
            target_top_priority = top_priorities[target_stack]

            min_priorities = torch.where(masked_x.squeeze() > 0, masked_x.squeeze(), n_containers*10).amin(dim=1)

            ideal_stacks = (min_priorities > target_top_priority) & valid_stacks

            if ideal_stacks.any():
                # Rule 1 (SSI tie randomly break)
                # get ssi value
                indices = torch.arange(len(min_priorities))
                ssi = min_priorities + self.pr * (indices % n_rows + 1) + self.pb * abs((indices // n_rows + 1) - (target_stack.squeeze() // n_rows + 1))

                # among ideal_stacks, find minimum ssi value stack
                valid_ssi = torch.where(ideal_stacks, ssi, float('inf'))
                min_indices = torch.where(valid_ssi == valid_ssi.min())[0]
                best_stack_index = min_indices[torch.randint(len(min_indices), (1,))]  # random tie break
                dest_index = best_stack_index.unsqueeze(1)

                if not restricted:
                    while True:
                        if max_tiers - stack_len[best_stack_index].squeeze() < 2:
                            break

                        valid_x = torch.where(masked_x.squeeze() > 0, masked_x.squeeze(), float('inf'))
                        valid_x[indices, stack_len - 1] = float('inf')

                        min_below_top = valid_x.amin(dim=1)

                        # Rule 2 (no tie)
                        candidate_stacks = (min_below_top < top_priorities)\
                                & (target_top_priority < top_priorities)\
                                & (min_priorities[best_stack_index] - 5 < top_priorities)\
                                & (top_priorities < min_priorities[best_stack_index])
                        
                        if candidate_stacks.any():
                            max_top_priority = top_priorities[candidate_stacks].max()
                            source_stack = torch.where(candidate_stacks & (top_priorities == max_top_priority))[0]

                            source_index = source_stack.unsqueeze(1)
                            cost += env.step(dest_index, source_index, no_clear=True)
                            self.check_validity(env.x.squeeze())

                            masked_x = env.x.clone()

                            if online:
                                mask = masked_x > online_container_num
                                for b in range(masked_x.shape[0]):
                                    masked_x[b][mask[b]] = online_container_num + 1
                            
                            stack_len = (masked_x > 0).sum(dim=2).squeeze()
                            top_priorities = self.get_top_priority(masked_x).squeeze()
                            min_priorities = torch.where(masked_x.squeeze() > 0, masked_x.squeeze(), n_containers*10).amin(dim=1)
                        else:
                            break

                source_index = target_stack.unsqueeze(1)
                cost += env.step(dest_index, source_index, no_clear=True)
                self.check_validity(env.x.squeeze())


            else:
                # Rule 3 (no tie)
                candidate_stacks = valid_stacks.clone()
                candidate_stacks[target_stack] = False

                max_min_priority = min_priorities[candidate_stacks].max()
                best_stack_index = torch.where(candidate_stacks & (min_priorities == max_min_priority))[0]

                if online:
                    if best_stack_index.shape[0] > 1:
                        curr_bay, curr_row =  (target_stack.item() // n_rows) + 1, (target_stack.item() % n_rows) + 1

                        chosen_idx, chosen_cost, (bay, row) = choose_stack_by_travel_time( # tie break by travel time
                            best_stack_index,
                            curr_bay=curr_bay, curr_row=curr_row,
                            n_rows=n_rows,
                            t_acc=env.t_acc, t_bay=env.t_bay, t_row=env.t_row,
                        )
                        best_stack_index = torch.tensor([chosen_idx])
                
                source_index = target_stack.unsqueeze(1)
                dest_index = best_stack_index.unsqueeze(1)
                cost += env.step(dest_index, source_index, no_clear=True)
                self.check_validity(env.x.squeeze())

        moves = n_containers + env.relocations.squeeze().item()
        return cost.squeeze().item(), moves


# """ online setting """
def choose_stack_by_travel_time(
    best_stack_index: torch.Tensor,
    curr_bay: int,
    curr_row: int,
    n_rows: int,
    t_acc: float, t_bay: float, t_row: float,
    device=None, generator: torch.Generator | None = None
):
    if device is None:
        device = best_stack_index.device

    idx = best_stack_index.to(device)

    bays = idx // n_rows + 1
    rows = idx % n_rows + 1

    curr_bay_t = torch.as_tensor(curr_bay, device=device)
    curr_row_t = torch.as_tensor(curr_row, device=device)

    bay_diff = (bays - curr_bay_t).abs()
    row_diff = (rows - curr_row_t).abs()

    bay_cost = torch.where(bay_diff != 0,
                           torch.as_tensor(t_acc, device=device, dtype=torch.float32) +
                           bay_diff.to(torch.float32) * torch.as_tensor(t_bay, device=device, dtype=torch.float32),
                           torch.tensor(0.0, device=device))

    row_cost = row_diff.to(torch.float32) * torch.as_tensor(t_row, device=device, dtype=torch.float32)
    cost = bay_cost + row_cost

    min_cost = cost.min()
    mask = (cost == min_cost)
    tie_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)

    pick_pos = torch.randint(0, tie_ids.numel(), (1,), device=device, generator=generator).item()
    chosen_pos = tie_ids[pick_pos].item()

    chosen_idx = idx[chosen_pos].item()
    chosen_cost = cost[chosen_pos].item()
    chosen_bay = bays[chosen_pos].item()
    chosen_row = rows[chosen_pos].item()

    return chosen_idx, chosen_cost, (chosen_bay, chosen_row)



if __name__ == "__main__":
    import time
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

                    input, inst_name = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

                    s = time.time()
                    lin = Lin2015() # batch X
                    wt, moves = lin.run(input)

                    print(f'inst_name: {inst_name}, cost: {wt}, time: {round(time.time()-s,1)}')

                    results.append([inst_name, wt, time.time()-s])
    
    import pandas as pd
    df = pd.DataFrame(results, columns=["inst_name", "WT", "C"])
    df.to_excel('./tmp_Lin.xlsx', index=False)
