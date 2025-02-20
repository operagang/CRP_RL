import time
import torch
import numpy as np
from torch.utils.data import Dataset

layout_to_n_containers = {
    (1,16,6)    :70,
    (2,16,6)    :140,
    (4,16,6)    :280,
    (6,16,6)    :430,
    (8,16,6)    :570,
    (10,16,6)   :720,
    (1,16,8)    :90,
    (2,16,8)    :190,
    (4,16,8)    :380,
    (6,16,8)    :570,

    (2,4,6)     :35
}

def get_n_containers(n_bays, n_rows, n_tiers):
    return layout_to_n_containers[n_bays, n_rows, n_tiers]



class Generator(Dataset):
    def __init__(self, args, seed=None, eval=False):
        if not eval:
            self.n_samples = args.batch_size * args.batch_num
            self.seed = seed
        else:
            self.n_samples = args.eval_batch_size * args.eval_batch_num
            self.seed = args.eval_seed
        self.n_stacks = args.n_bays * args.n_rows  # 전체 stack 개수
        self.n_tiers = args.n_tiers
        self.n_containers = get_n_containers(args.n_bays, args.n_rows, args.n_tiers)
        self.instance_type = args.instance_type
        self.n_bays = args.n_bays
        self.n_rows = args.n_rows
        

        # ✅ 데이터 생성
        self.data = self.generate_data()

    def generate_data(self):
        """
        랜덤한 컨테이너 배치를 생성하고, instance_type이 'upsidedown'이면 각 stack을 정렬.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        clock = time.time()
        data = torch.zeros((self.n_samples, self.n_stacks, self.n_tiers), dtype=torch.float32)  # ✅ FloatTensor 사용

        for i in range(self.n_samples):
            # 1~n_containers 의 수를 랜덤한 순서로 생성 (float으로 변환)
            container_sequence = (np.random.permutation(self.n_containers) + 1).astype(np.float32)  

            # ✅ 컨테이너를 하나씩 랜덤한 스택에 배치
            stack_fill_counts = np.zeros(self.n_stacks, dtype=int)  # 각 stack에 현재 채워진 개수

            for container in container_sequence:
                valid_stacks = np.where(stack_fill_counts < self.n_tiers)[0]  # 아직 공간이 있는 stack 선택
                if len(valid_stacks) == 0 and stack_fill_counts.sum() != self.n_containers:
                    raise ValueError('stack이 가득 참')
                
                selected_stack = np.random.choice(valid_stacks)  # 랜덤한 stack 선택
                tier_position = stack_fill_counts[selected_stack]  # 해당 stack의 채울 위치
                data[i, selected_stack, tier_position] = torch.tensor(container, dtype=torch.float32)  # 컨테이너 배치 (float32)
                stack_fill_counts[selected_stack] += 1  # 해당 stack에 채운 개수 증가

            # ✅ instance_type이 'upsidedown'이면 각 stack을 오름차순 정렬
            if self.instance_type == 'upsidedown':
                for s in range(self.n_stacks):
                    nonzero_values = data[i, s][data[i, s] > 0]  # 0이 아닌 값만 추출
                    sorted_values = torch.sort(nonzero_values)[0]  # 0이 아닌 값만 정렬
                    data[i, s, :len(sorted_values)] = sorted_values  # 정렬된 값 삽입, 나머지는 그대로 유지
            elif self.instance_type == 'random':
                pass
            else:
                raise ValueError('instance type 입력이 잘못됨')
        print(f'{self.n_samples}개 data 생성시간: {round(time.time() - clock, 2)}초')
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index]



if __name__ == '__main__':
    dataset = Generator(args)
