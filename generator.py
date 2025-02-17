import torch
import numpy as np
from torch.utils.data import Dataset

class Generator(Dataset):
    def __init__(self, n_samples, n_bays, n_rows, n_tiers, n_containers, instance_type):
        self.n_samples = n_samples
        self.n_stacks = n_bays * n_rows  # 전체 stack 개수
        self.n_tiers = n_tiers
        self.n_containers = n_containers
        self.instance_type = instance_type

        # ✅ 데이터 생성
        self.data = self.generate_data()

    def generate_data(self):
        """
        랜덤한 컨테이너 배치를 생성하고, instance_type이 'upsidedown'이면 각 stack을 정렬.
        """
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

        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index]



if __name__ == '__main__':
    dataset = Generator(n_samples=5, n_bays=2, n_rows=3, n_tiers=4, n_containers=15, instance_type='upsidedown')
