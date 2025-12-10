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



class Kim2016(): # batch 연산 X
    def __init__(self):
        pass

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

    def find_critical_stacks(self, x):
        num_stacks, max_tiers = x.shape  # (stack, tiers)

        # 0이 아닌 컨테이너 위치 찾기
        mask = x > 0  

        # 각 컨테이너의 위쪽(index가 큰) 컨테이너 찾기
        above_mask = torch.triu(torch.ones((max_tiers, max_tiers), dtype=torch.bool, device=x.device), diagonal=1)  # 상삼각 행렬
        above_containers = mask.unsqueeze(1) & above_mask  # 자기보다 위쪽에 있는 컨테이너들

        # 자기보다 큰 컨테이너 찾기
        larger_than_self = (x.unsqueeze(1) > x.unsqueeze(2)) & above_containers  # 자신보다 위쪽에 있으면서 더 큰 priority

        # 위쪽 컨테이너들의 개수 계산
        num_above_containers = above_containers.sum(dim=2)  # 자기보다 index가 큰 컨테이너 개수

        # 위쪽 컨테이너 중 자기보다 큰 priority를 가진 개수 계산
        num_larger_above_containers = larger_than_self.sum(dim=2)

        # Critical Container 판별: 위쪽의 자기보다 큰 컨테이너 개수 == 위쪽 컨테이너 개수
        is_critical_container = mask & (num_larger_above_containers == num_above_containers)

        # ✅ 각 stack에서 top container 찾기 (각 stack에서 가장 위에 있는 0이 아닌 값)
        top_indices = mask.int().cumsum(dim=1).argmax(dim=1)  # 각 stack의 top container 위치
        top_mask = torch.zeros_like(mask, dtype=torch.bool)  # top 위치를 위한 mask
        top_mask[torch.arange(num_stacks), top_indices] = True  # top container 위치에 True 설정

        # ✅ top container는 critical container가 될 수 없음
        is_critical_container = is_critical_container & ~top_mask

        # Critical Stack 찾기: 하나라도 Critical Container를 포함하는 stack
        critical_stacks = is_critical_container.any(dim=1)

        return critical_stacks

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

            # Ideal stack 찾기
            stack_len = (env.x > 0).sum(dim=2).squeeze()
            valid_stacks = stack_len < max_tiers
            is_non_increasing = env.x[:, :, :-1] >= env.x[:, :, 1:]
            stack_is_ideal = is_non_increasing.all(dim=2).squeeze() & valid_stacks

            if stack_is_ideal.any():
                top_priorities = self.get_top_priority(env.x).squeeze()
                target_top_priority = top_priorities[target_stack]  # (batch,)
                candidate_stacks = stack_is_ideal & (top_priorities > target_top_priority)

                if candidate_stacks.any():
                    # Case 3 (selected_stack의 tie-breaking: random, 나머지 tie 없음)
                    min_indices = torch.where(candidate_stacks & (top_priorities == top_priorities[candidate_stacks].min()))[0]
                    selected_stack = min_indices[torch.randint(len(min_indices), (1,))]
                    selected_top_priority = top_priorities[selected_stack]
                    dest_index = selected_stack.unsqueeze(1)

                    if not restricted:
                        while True:
                            if max_tiers - stack_len[selected_stack].squeeze() < 2:
                                break

                            critical_stacks = self.find_critical_stacks(env.x.squeeze())
                            # ✅ 1. critical stack 중에서 target_top_priority < priority < selected_top_priority인 stack 선택
                            selected_critical_stacks = critical_stacks & (top_priorities > target_top_priority) & (top_priorities < selected_top_priority)

                            # ✅ 2. 선택된 stack 중 top priority가 가장 큰 stack index 찾기
                            if selected_critical_stacks.any():
                                max_priority = top_priorities[selected_critical_stacks].max()  # 선택된 stack 중 top priority 최대값
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
                    # Case 4 (tie 없음)
                    selected_stack = torch.where(stack_is_ideal & (top_priorities == top_priorities[stack_is_ideal].max()))[0]
                    selected_top_priority = top_priorities[selected_stack]
                    dest_index = selected_stack.unsqueeze(1)

                    if not restricted:
                        while True:
                            if max_tiers - stack_len[selected_stack].squeeze() < 2:
                                break

                            critical_stacks = self.find_critical_stacks(env.x.squeeze())
                            # ✅ 1. critical stack 중에서 target_top_priority < priority < selected_top_priority인 stack 선택
                            selected_critical_stacks = critical_stacks & (top_priorities < selected_top_priority)

                            # ✅ 2. 선택된 stack 중 top priority가 가장 큰 stack index 찾기
                            if selected_critical_stacks.any():
                                max_priority = top_priorities[selected_critical_stacks].max()  # 선택된 stack 중 top priority 최대값
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
                # Case 2 (tie 없음)
                min_prios = torch.min(torch.where(env.x.squeeze() == .0, torch.FloatTensor([1+num_stacks*max_tiers]), env.x.squeeze()), dim=1)[0]
                max_min_prio = min_prios[valid_stacks].max()  # 선택된 stack 중 min priority 최대값
                dest_index = torch.where(valid_stacks & (min_prios == max_min_prio))[0].unsqueeze(1)
                source_index = target_stack.unsqueeze(1)
                cost += env.step(dest_index, source_index, no_clear=True)
                self.check_validity(env.x.squeeze())

        moves = n_containers + env.relocations.squeeze().item()
        return cost.squeeze().item(), moves



if __name__ == "__main__":
    from benchmarks.benchmarks import find_and_process_file
    import time
    
    folder_path = "./benchmarks/Shin_instances"  # Replace with the folder containing your files
    n_rows = 16
    results = []
    for inst_type in ['random', 'upsidedown']:
        for n_tiers in [6,8]:
            for n_bays in [20,30]:
                for id in range(1,21):

                    input, inst_name = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)

                    s = time.time()
                    lin = Kim2016() # batch 연산 X
                    wt, moves = lin.run(input)

                    print(f'inst_name: {inst_name}, cost: {wt}, time: {round(time.time()-s,1)}')

                    results.append([inst_name, wt, time.time()-s])
    
    import pandas as pd
    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["inst_name", "WT", "C"])
    
    # 엑셀 파일로 저장
    df.to_excel('./tmp_kim.xlsx', index=False)