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

def count_disorder_per_row(x):
    max_tiers = x.shape[-1]

    # 비교 행렬: i > j인 인덱스 마스크 (아래쪽 인덱스만)
    i = torch.arange(max_tiers, device=x.device).view(1, -1, 1)  # shape (1, S, 1)
    j = torch.arange(max_tiers, device=x.device).view(1, 1, -1)  # shape (1, 1, S)
    mask = j < i  # shape (1, S, S), True where j < i (i는 위, j는 아래)

    x_expanded = x.unsqueeze(2)  # shape (R, S, 1)
    x_below = x.unsqueeze(1)     # shape (R, 1, S)

    valid = (x_expanded != 0) & (x_below != 0) & mask  # 유효 비교 위치만

    compare = (x_expanded > x_below) & valid  # 위 > 아래 인 경우만

    disorder_flag = compare.any(dim=2)  # shape (R, S), 각 위치에 대해 아래에 더 작은 게 있는지

    count = disorder_flag.sum(dim=1)  # row별 합산
    return count


def get_wt_lb(x):
    n_bays = x.shape[1]
    n_rows = x.shape[2]
    t_pd = 30
    t_acc = 40
    t_bay = 3.5
    t_row = 1.2


    x = x.squeeze(0)               # [2, 16, 6]
    x = x.view(-1, x.size(-1))  # [32, 6]

    curr_bay = None
    curr_row = None
    
    # 0이 아닌 값 위치 (row, col)
    nonzero_pos = (x != 0).nonzero(as_tuple=False)

    # if nonzero_pos.numel() == 0:
    #     return 0.0  # 혹시 아무 컨테이너도 없을 경우 안전 처리

    # 그 위치의 값과 행 인덱스를 가져옴
    values = x[nonzero_pos[:, 0], nonzero_pos[:, 1]]
    stacks = nonzero_pos[:, 0]

    # 값 기준으로 정렬 (작은 값부터)
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
    from benchmarks.benchmarks import find_and_process_file

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

                    cost = get_wt_lb(container_tensor) # batch 연산 X

                    print(f'inst_name: {inst_name}')
                    print(f'cost: {cost}')

                    results.append([inst_name, cost])
    
    import pandas as pd
    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["inst_name", "WT"])
    
    # 엑셀 파일로 저장
    df.to_excel('./tmp.xlsx', index=False)