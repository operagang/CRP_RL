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



class Durasevic2025():
    def __init__(self):
        self.empty_avg = 'N+1' # 'N+1' or '0'
        self.empty_min = 'N+1' # 'N+1' or '0' or 'target_top'
        self.obj = 'WT' # WT or reloc
        self.restricted = False

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

    def run(self, x):
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

            slack = n_containers - env.x.max()
            tmp_x = torch.where(env.x > 0, env.x + slack, 0)

            SH = (tmp_x > 0).sum(dim=2).squeeze()
            EMP = max_tiers - SH

            top_priorities = self.get_top_priority(tmp_x).squeeze()
            target_stack = self.get_target_stack(tmp_x)
            CUR = top_priorities[target_stack] # target top priority

            RI = ((tmp_x > 0) & (tmp_x < CUR)).sum(dim=2).squeeze()
            mask = tmp_x > 0
            AVG = (tmp_x * mask).sum(dim=2) / mask.sum(dim=2)
            AVG = AVG.masked_fill(torch.isnan(AVG), 0.0)

            if self.empty_avg == 'N+1':
                AVG = torch.where(AVG > 0, AVG, tmp_x.max() + 1)
            elif self.empty_avg == '0':
                pass
            else:
                raise ValueError('wrong self.empty_avg')

            min_priorities = torch.where(tmp_x > 0, tmp_x, float('inf')).amin(dim=2).squeeze()

            if self.empty_min == '0':
                min_priorities = torch.where(min_priorities != float('inf'), min_priorities, 0)
            elif self.empty_min == 'N+1':
                min_priorities = torch.where(min_priorities != float('inf'), min_priorities, tmp_x.max() + 1)
            elif self.empty_min == 'target_top':
                min_priorities = torch.where(min_priorities != float('inf'), min_priorities, CUR.item())
            else:
                raise ValueError('wrong self.empty_min')

            DIFF = min_priorities - CUR.item()

            indices = torch.arange(len(min_priorities))
            DIS = abs((indices // n_rows + 1) - (target_stack.squeeze() // n_rows + 1))
            DUR = DIS * env.t_bay + (DIS > 0) * env.t_acc


            eps = 1e-9
            if self.obj == 'WT':
                PF = ((((RI + DIS) * ((DIFF * DIFF) + (DUR - DIFF))) * RI)
                    + (((RI - DUR) / (DUR - (SH / (DIFF + eps)) + eps))
                    / ((DUR / (CUR / (DUR + eps) + eps)) * ((DIFF * DIFF) + (DUR * CUR)) + eps)))
            elif self.obj == 'reloc':
                PF = ((((RI + DIS) * ((DIFF * DIFF) + (DUR - EMP))) * RI)
                    + (((((DUR - EMP) - (DUR + RI)) / ((DIFF * EMP) - (SH / (DIS + eps)) + eps))
                    / (((CUR + RI) / (EMP + eps)) * ((DIFF * DIFF) + DUR) + eps))))

            valid_stacks = SH < max_tiers
            valid_stacks[target_stack] = False
            valid_indices = torch.where(valid_stacks)[0]
            valid_PF_values = PF[valid_indices]
            min_PF_value = torch.min(valid_PF_values)
            min_indices = valid_indices[torch.where(valid_PF_values == min_PF_value)[0]]
            best_stack_index = min_indices[torch.randint(len(min_indices), (1,))]  # 랜덤 선택
            dest_index = best_stack_index.unsqueeze(1)

            if not self.restricted:
                SC = CUR.item()
                DC = top_priorities[best_stack_index].item()
                while True:
                    if SC <= DC:
                        break
                    min_prio = torch.where(tmp_x > 0, tmp_x, float('inf')).amin(dim=2).squeeze()
                    stack_height = (tmp_x > 0).sum(dim=2).squeeze()
                    is_candidate = (min_prio > DC) & (stack_height < max_tiers)
                    is_candidate[target_stack] = False

                    if not is_candidate.sum() > 0:
                        break

                    candidate_stacks = torch.where(is_candidate)[0]
                    new_dest_idx = candidate_stacks[torch.randint(len(candidate_stacks), (1,))].unsqueeze(1)
                    
                    cost += env.step(new_dest_idx, dest_index, no_clear=True)
                    self.check_validity(env.x.squeeze())

                    tmp_x = torch.where(env.x > 0, env.x + slack, 0)
                    top_priorities = self.get_top_priority(tmp_x).squeeze()
                    DC = top_priorities[best_stack_index].item()

            source_index = target_stack.unsqueeze(1)
            cost += env.step(dest_index, source_index, no_clear=True)
            self.check_validity(env.x.squeeze())



        moves = n_containers + env.relocations.squeeze().item()
        return cost.squeeze().item(), moves




if __name__ == "__main__":
    from benchmarks.benchmarks import find_and_process_file
    import time

    # # Example usage
    # folder_path = "./benchmarks/Lee_instances"  # Replace with the folder containing your files
    # inst_type = "random"
    # n_bays = 2
    # n_rows = 16
    # n_tiers = 6
    # id = 3

    # container_tensor, _ = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

    # arg = Durasevic2025() # batch 연산 X

    # cost = arg.run(container_tensor)

    # print(cost)


    # Example usage
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

                    s = time.time()
                    arg = Durasevic2025() # batch 연산 X

                    cost = arg.run(container_tensor)

                    print(f'inst_name: {inst_name}')
                    print(f'cost: {cost}')

                    results.append([inst_name, cost, time.time()-s])
    
    import pandas as pd
    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["inst_name", "WT", "C"])
    
    # 엑셀 파일로 저장
    df.to_excel('./tmp.xlsx', index=False)