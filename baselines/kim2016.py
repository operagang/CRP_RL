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



class Kim2016(): # batch X
    def __init__(self):
        pass

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

    def find_critical_stacks(self, x):
        num_stacks, max_tiers = x.shape  # (stack, tiers)

        mask = x > 0  

        above_mask = torch.triu(torch.ones((max_tiers, max_tiers), dtype=torch.bool, device=x.device), diagonal=1)
        above_containers = mask.unsqueeze(1) & above_mask

        larger_than_self = (x.unsqueeze(1) > x.unsqueeze(2)) & above_containers

        num_above_containers = above_containers.sum(dim=2)

        num_larger_above_containers = larger_than_self.sum(dim=2)

        is_critical_container = mask & (num_larger_above_containers == num_above_containers)

        top_indices = mask.int().cumsum(dim=1).argmax(dim=1)
        top_mask = torch.zeros_like(mask, dtype=torch.bool)
        top_mask[torch.arange(num_stacks), top_indices] = True

        is_critical_container = is_critical_container & ~top_mask

        critical_stacks = is_critical_container.any(dim=1)

        return critical_stacks

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

    def run(self, x, restricted=False):
        _, n_bays, n_rows, max_tiers = x.shape  # (1, bays, rows, tiers)
        num_stacks = n_bays * n_rows
        n_containers = torch.sum(x > 0).item()
        device = torch.device('cpu')

        env = Env(device, x)
        cost = torch.zeros(env.x.shape[0])

        while True:
            # Case 1
            cost += env.clear()

            self.check_validity(env.x.squeeze())

            if env.all_empty():
                break

            target_stack = self.get_target_stack(env.x)

            # find ideal stack
            stack_len = (env.x > 0).sum(dim=2).squeeze()
            valid_stacks = stack_len < max_tiers
            is_non_increasing = env.x[:, :, :-1] >= env.x[:, :, 1:]
            stack_is_ideal = is_non_increasing.all(dim=2).squeeze() & valid_stacks

            if stack_is_ideal.any():
                top_priorities = self.get_top_priority(env.x).squeeze()
                target_top_priority = top_priorities[target_stack]  # (batch,)
                candidate_stacks = stack_is_ideal & (top_priorities > target_top_priority)

                if candidate_stacks.any():
                    # Case 3 (selected_stack tie-breaking: random, no further tie exists)
                    min_indices = torch.where(candidate_stacks & (top_priorities == top_priorities[candidate_stacks].min()))[0]
                    selected_stack = min_indices[torch.randint(len(min_indices), (1,))]
                    selected_top_priority = top_priorities[selected_stack]
                    dest_index = selected_stack.unsqueeze(1)

                    if not restricted:
                        while True:
                            if max_tiers - stack_len[selected_stack].squeeze() < 2:
                                break

                            critical_stacks = self.find_critical_stacks(env.x.squeeze())
                            # 1. among critical stacks, "target_top_priority < priority < selected_top_priority" stack selection
                            selected_critical_stacks = critical_stacks & (top_priorities > target_top_priority) & (top_priorities < selected_top_priority)

                            # 2. among selected stacks, find the stack with the highest top priority
                            if selected_critical_stacks.any():
                                max_priority = top_priorities[selected_critical_stacks].max()
                                source_index = torch.where(selected_critical_stacks & (top_priorities == max_priority))[0].unsqueeze(1)
                                cost += env.step(dest_index, source_index, no_clear=True)
                                self.check_validity(env.x.squeeze())
                                stack_len = (env.x > 0).sum(dim=2).squeeze()
                                top_priorities = self.get_top_priority(env.x).squeeze()
                                selected_top_priority = top_priorities[selected_stack]
                            else:
                                break
                    
                    source_index = target_stack.unsqueeze(1)
                    cost += env.step(dest_index, source_index, no_clear=True)
                    self.check_validity(env.x.squeeze())

                else:
                    # Case 4 (no tie)
                    selected_stack = torch.where(stack_is_ideal & (top_priorities == top_priorities[stack_is_ideal].max()))[0]
                    selected_top_priority = top_priorities[selected_stack]
                    dest_index = selected_stack.unsqueeze(1)

                    if not restricted:
                        while True:
                            if max_tiers - stack_len[selected_stack].squeeze() < 2:
                                break

                            critical_stacks = self.find_critical_stacks(env.x.squeeze())
                            # 1. among critical stacks, "target_top_priority < priority < selected_top_priority" stack selection
                            selected_critical_stacks = critical_stacks & (top_priorities < selected_top_priority)

                            # 2. among selected stacks, find the stack with the highest top priority
                            if selected_critical_stacks.any():
                                max_priority = top_priorities[selected_critical_stacks].max()
                                source_index = torch.where(selected_critical_stacks & (top_priorities == max_priority))[0].unsqueeze(1)
                                cost += env.step(dest_index, source_index, no_clear=True)
                                self.check_validity(env.x.squeeze())
                                stack_len = (env.x > 0).sum(dim=2).squeeze()
                                top_priorities = self.get_top_priority(env.x).squeeze()
                                selected_top_priority = top_priorities[selected_stack]
                            else:
                                break
                    
                    source_index = target_stack.unsqueeze(1)
                    cost += env.step(dest_index, source_index, no_clear=True)
                    self.check_validity(env.x.squeeze())

            else:
                # Case 2 (no tie)
                min_prios = torch.min(torch.where(env.x.squeeze() == .0, torch.FloatTensor([1+num_stacks*max_tiers]), env.x.squeeze()), dim=1)[0]
                max_min_prio = min_prios[valid_stacks].max()
                dest_index = torch.where(valid_stacks & (min_prios == max_min_prio))[0].unsqueeze(1)
                source_index = target_stack.unsqueeze(1)
                cost += env.step(dest_index, source_index, no_clear=True)
                self.check_validity(env.x.squeeze())

        moves = n_containers + env.relocations.squeeze().item()
        return cost.squeeze().item(), moves



if __name__ == "__main__":
    from benchmarks.benchmarks import find_and_process_file
    import time
    
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
                    lin = Kim2016() # batch X
                    wt, moves = lin.run(input)

                    print(f'inst_name: {inst_name}, cost: {wt}, time: {round(time.time()-s,1)}')

                    results.append([inst_name, wt, time.time()-s])
    
    import pandas as pd
    df = pd.DataFrame(results, columns=["inst_name", "WT", "C"])
    df.to_excel('./tmp_kim.xlsx', index=False)