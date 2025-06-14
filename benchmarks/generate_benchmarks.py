import torch
try:
    from generator.generator import Generator
except:
    import os, sys
    # 현재 스크립트(`lin2015.py`)가 위치한 디렉터리
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 최상위 디렉터리 경로 (현재 스크립트보다 한 단계 위)
    top_level_dir = os.path.abspath(os.path.join(current_dir, ".."))
    # 최상위 디렉터리를 sys.path에 추가 (실행 중에만 적용됨)
    if top_level_dir not in sys.path:
        sys.path.append(top_level_dir)
    from generator.generator import Generator


def save_data(data, save_path, inst_type, n_bays, n_rows, n_tiers, n_containers, target_id):
    # 이름 구성
    bays_str = f"{'R' if inst_type == 'random' else 'U'}{n_bays:02d}"
    stacks_str = f"{n_rows:02d}"
    tiers_str = f"{n_tiers:02d}"
    containers_str = f"{n_containers:04d}"
    id_str = f"{target_id:03d}"
    problem_name = f"{bays_str}{stacks_str}{tiers_str}_{containers_str}_{id_str}"

    if inst_type == 'random':
        save_path = f'{save_path}/individual, random'
    else:
        save_path = f'{save_path}/individual, upside down'

    # 파일로 저장
    with open(f"{save_path}/{problem_name}.txt", "w") as f:
        # 헤더 라인
        f.write(f"{problem_name} {n_bays} {n_rows} {n_tiers} {n_containers} {n_containers}\n")
        
        for b in range(n_bays):
            for r in range(n_rows):
                values = data[b, r]
                nonzero_values = values[values > 0]
                if len(nonzero_values) == 0:
                    continue
                line = f"{b+1:2d} {r+1:2d} {len(nonzero_values):2d}"
                for v in nonzero_values:
                    val = int(v.item())
                    line += f" {val:3d} {val:3d}"
                f.write(line + "\n")



if __name__=='__main__':
    save_path = './benchmarks/New_instances'
    
    n_rows = 16
    for n_bays in [30]:
        for n_tiers in [6,8]:
            for inst_type in ['random', 'upsidedown']:
                for target_id in range(1,21):
                    n_containers = int(0.75 * n_bays * n_tiers * n_rows)

                    data = Generator(
                        n_samples=1,
                        layout=(n_containers, n_bays, n_rows, n_tiers),
                        inst_type=inst_type,
                        device='cpu'
                    )[0]

                    save_data(data, save_path, inst_type, n_bays, n_rows, n_tiers, n_containers, target_id)

