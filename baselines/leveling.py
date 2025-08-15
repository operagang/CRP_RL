import torch
try:
    from env.env import Env
except:
    import os, sys
    # 현재 스크립트(`lin2015.py`)가 위치한 디렉터리
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 최상위 디렉터리 경로 (현재 스크립트보다 한 단계 위)
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    # 최상위 디렉터리를 sys.path에 추가 (실행 중에만 적용됨)
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from env.env import Env



class Leveling():
    def __init__(self, pr=30, pb=300):
        self.pr = pr # SSI에 row 관련 weight
        self.pb = pb # SSI에 bay 관련 weight

        """"""
        self.bay_diff = []
        self.row_diff = []
        self.well_located = []
    
    """"""
    def save_log(self, srce_idxs, dest_idxs, env):
        srce_idxs = srce_idxs[0]
        dest_idxs = dest_idxs[0]
        n_bays = env.n_bays
        n_rows = env.n_rows
        for i in range(env.empty.shape[0]):
            if not env.empty[i]:
                s_bay = srce_idxs[i] // n_rows + 1
                s_row = srce_idxs[i] % n_rows + 1
                d_bay = dest_idxs[i] // n_rows + 1
                d_row = dest_idxs[i] % n_rows + 1
                self.bay_diff.append(abs(s_bay-d_bay).item())
                self.row_diff.append(abs(s_row-d_row).item())

                top = env.x[i][srce_idxs[i]][(env.x[i][srce_idxs[i]] != 0).nonzero(as_tuple=True)[0][-1]]
                d_stack = env.x[i][dest_idxs[i]].clone()
                d_stack[d_stack == 0] = 100000
                self.well_located.append((top < d_stack.min()).item())

    def get_top_priority(self, x):
        _, num_stacks, max_tiers = x.shape  # (batch, stack, tiers)
        # 0이 아닌 값들을 1로 변환하고, stack의 실제 높이 계산
        top_idxs = (x > 0).sum(dim=2) - 1  # 각 stack의 마지막 컨테이너 인덱스
        # stack이 비어있는 경우, 대체값(num_stacks * max_tiers * 10) 사용
        top_priorities = torch.where(
            top_idxs >= 0,  # 유효한 stack인지 확인
            x[torch.arange(1).view(-1, 1), torch.arange(num_stacks).view(1, -1), top_idxs], 
            torch.tensor(num_stacks * max_tiers * 10, device=x.device, dtype=x.dtype)
        )
        return top_priorities

    def get_target_stack(self, x):
        _, num_stacks, max_tiers = x.shape  # (batch, stack, tiers)
        mn_val = torch.min(torch.where(x == .0, torch.FloatTensor([1+num_stacks*max_tiers]), x), dim=2)[0]
        target_stack = torch.argmin(mn_val, dim=1) # mn_val: stack 별 최소값
        return target_stack

    def check_validity(self, x):
        # ✅ 1. 전체에서 0이 아닌 값들 찾기
        nonzero_values = x[x > 0].int().tolist()  # 전체에서 0이 아닌 값들 리스트로 저장
        n = len(nonzero_values)  # 0이 아닌 값들의 전체 개수

        # ✅ 2. `1~n`까지의 값이 정확히 하나씩 존재하는지 확인
        required_values = set(range(1, n + 1))  # {1, 2, ..., n}
        is_valid = set(nonzero_values) == required_values  # 필요한 값들이 모두 존재하는지 확인

        # ✅ 3. 각 stack이 "첫 번째 0이 아닌 값 이후에는 반드시 0이어야 하는지" 체크
        for stack in x:
            nonzero_idxs = (stack > 0).nonzero(as_tuple=True)[0]  # 첫 번째 0이 아닌 값의 위치 찾기
            if len(nonzero_idxs) > 0:  # 0이 아닌 값이 존재하는 경우만 체크
                last_nonzero_idx = nonzero_idxs[-1].item()  # 첫 번째 등장 index
                if (stack[last_nonzero_idx + 1:] > 0).any():  # 이후 값 중 0이 아닌 값이 존재하면 False
                    is_valid = False
                    break  # 하나라도 위반하면 바로 종료

        if not is_valid:
            raise ValueError(f"not valid env.x: \n{x}")

        return

    def run(self, x, restricted=False):
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

            target_stack = self.get_target_stack(env.x)
            
            stack_len = (env.x > 0).sum(dim=2).squeeze()
            valid_stacks = stack_len < max_tiers
            valid_stacks[target_stack] = False

            target_bay = target_stack // n_rows  # 0-based bay index
            bay_mask = (torch.arange(stack_len.size(0), device=stack_len.device) // n_rows) == target_bay

            # 같은 bay 중 valid만
            same_bay_valid = valid_stacks & bay_mask

            if same_bay_valid.any():
                # 같은 bay에서만 선택
                min_len = stack_len[same_bay_valid].min()
                best_stack_index = torch.nonzero((stack_len == min_len) & same_bay_valid, as_tuple=False).squeeze(1)
            else:
                # 기존 전체 valid에서 선택
                min_len = stack_len[valid_stacks].min()
                best_stack_index = torch.nonzero((stack_len == min_len) & valid_stacks, as_tuple=False).squeeze(1)

            if best_stack_index.shape[0] > 1:
                curr_bay, curr_row =  (target_stack.item() // n_rows) + 1, (target_stack.item() % n_rows) + 1  # target에서 현재위치 정할거면 이런 식으로
                # 혹은 현재 크레인 위치가 따로 관리된다면 그걸 넣으세요.

                chosen_idx, chosen_cost, (bay, row) = choose_stack_by_travel_time(
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

            """"""
            self.save_log(source_index, dest_index, env)

        moves = n_containers + env.relocations.squeeze().item()
        return cost.squeeze().item(), moves




def choose_stack_by_travel_time(
    best_stack_index: torch.Tensor,   # 예: tensor([ 3, 11, 12])
    curr_bay: int,                    # 현재 크레인 bay (1-based)
    curr_row: int,                    # 현재 크레인 row (1-based)
    n_rows: int,                      # 한 bay당 row 수
    t_acc: float, t_bay: float, t_row: float,
    device=None, generator: torch.Generator | None = None
):
    """
    1) best_stack_index 후보들의 (bay,row) 계산
    2) 이동시간 cost = (bay 다르면 t_acc + |Δbay|*t_bay, 같으면 0) + |Δrow|*t_row
    3) 최소 cost 중 랜덤 tie-break
    반환: (선택된 stack_idx, 해당 cost, (bay,row))
    """
    if device is None:
        device = best_stack_index.device

    idx = best_stack_index.to(device)

    # 1-based bay/row로 변환
    bays = idx // n_rows + 1
    rows = idx % n_rows + 1

    curr_bay_t = torch.as_tensor(curr_bay, device=device)
    curr_row_t = torch.as_tensor(curr_row, device=device)

    # 이동시간 계산
    bay_diff = (bays - curr_bay_t).abs()
    row_diff = (rows - curr_row_t).abs()

    bay_cost = torch.where(bay_diff != 0,
                           torch.as_tensor(t_acc, device=device, dtype=torch.float32) +
                           bay_diff.to(torch.float32) * torch.as_tensor(t_bay, device=device, dtype=torch.float32),
                           torch.tensor(0.0, device=device))

    row_cost = row_diff.to(torch.float32) * torch.as_tensor(t_row, device=device, dtype=torch.float32)
    cost = bay_cost + row_cost  # 총 이동시간

    # 최소 cost 후보들 중 랜덤 선택
    min_cost = cost.min()
    mask = (cost == min_cost)
    tie_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # 균등확률로 하나 뽑기 (재현성 원하면 generator 전달)
    pick_pos = torch.randint(0, tie_ids.numel(), (1,), device=device, generator=generator).item()
    chosen_pos = tie_ids[pick_pos].item()

    chosen_idx = idx[chosen_pos].item()
    chosen_cost = cost[chosen_pos].item()
    chosen_bay = bays[chosen_pos].item()
    chosen_row = rows[chosen_pos].item()

    return chosen_idx, chosen_cost, (chosen_bay, chosen_row)



if __name__ == "__main__":
    """Option 1"""
    # # Example usage
    # from benchmarks.benchmarks import find_and_process_file
    # folder_path = "./benchmarks/Lee_instances"  # Replace with the folder containing your files
    # inst_type = "random"
    # n_bays = 2
    # n_rows = 16
    # n_tiers = 6
    # id = 3

    # input, _ = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

    # lin = Lin2015() # batch 연산 X
    # cost = lin.run(input)
    # print(cost)
    """"""


    # """Option 2"""
    # import torch
    # inputs = torch.load('./results/20250306_174550/eval_data.pt')

    # avg_wt, avg_moves = 0, 0
    # for i in range(inputs.shape[0]):
    #     input = inputs[i:i+1]
    #     lin = Lin2015()
    #     wt, moves = lin.run(input, restricted=False)
    #     avg_wt += wt
    #     avg_moves += moves
    #     print(i, wt, moves)
    
    # avg_wt /= inputs.shape[0]
    # avg_moves /= inputs.shape[0]

    # print(avg_wt, avg_moves)
    # """"""

    # """"""
    # data_by_instance = {}


    import time
    from benchmarks.benchmarks import find_and_process_file
    # Example usage
    folder_path = "./benchmarks/Lee_instances"  # Replace with the folder containing your files
    n_rows = 16
    results = []
    for inst_type in ['random', 'upsidedown']:
    # for inst_type in ['upsidedown']:
        for n_tiers in [6,8]:
            for n_bays in [1,2,4,6,8,10]:
                for id in range(1,6):
                    if n_tiers == 8 and n_bays in [8, 10]:
                        continue
                    if inst_type == 'upsidedown' and id in [3,4,5]:
                        continue
    # folder_path = "./benchmarks/Shin_instances"  # Replace with the folder containing your files
    # n_rows = 16
    # results = []
    # for inst_type in ['random', 'upsidedown']:
    #     for n_tiers in [6,8]:
    #         for n_bays in [20,30]:
    #             for id in range(1,21):

                    input, inst_name = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

                    s = time.time()
                    level = Leveling() # batch 연산 X
                    wt, moves = level.run(input)

                    print(f'inst_name: {inst_name}, cost: {wt}, time: {round(time.time()-s,1)}')

                    results.append([inst_name, wt, time.time()-s])


                    # """"""
                    # if inst_name[:-8] not in data_by_instance:
                    #     data_by_instance[inst_name[:-8]] = {
                    #         'well_located':[],
                    #         'bay_diff':[],
                    #         'row_diff':[]
                    #     }
                    # data_by_instance[inst_name[:-8]]['well_located'].extend(lin.well_located)
                    # data_by_instance[inst_name[:-8]]['bay_diff'].extend(lin.bay_diff)
                    # data_by_instance[inst_name[:-8]]['row_diff'].extend(lin.row_diff)

                    pass
                    

                    



    
    import pandas as pd
    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["inst_name", "WT", "C"])
    
    # 엑셀 파일로 저장
    df.to_excel('./tmp_Leveling.xlsx', index=False)




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
    # df_summary.to_excel("log_lin.xlsx", index=False)



    