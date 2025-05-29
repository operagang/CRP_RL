import time
import torch
import numpy as np
from torch.utils.data import Dataset



class Generator(Dataset):
    def __init__(self, load_data=None, seed=None, n_samples=None, layout=None, inst_type=None, device=None):
        # ✅ 데이터 생성
        if load_data is None:            
            self.n_samples = n_samples
            self.data = self.generate_data_vectorized(seed, layout, inst_type, device)
        else:
            self.data = torch.load(load_data)
            self.n_samples = self.data.shape[0]


    def generate_data_vectorized(self, seed, layout, inst_type, device):
        """
        완전히 PyTorch 기반으로 GPU 연산을 수행하여 컨테이너 배치를 생성
        """
        n_containers, n_bays, n_rows, n_tiers = layout
        n_stacks = n_bays * n_rows
        assert n_containers <= n_bays * n_rows * n_tiers
        if seed is not None:
            torch.manual_seed(seed)  # ✅ PyTorch 난수 시드 고정

        clock = time.time()

        # ✅ PyTorch Tensor를 사용하여 초기화 (GPU로 이동 가능)
        data = torch.zeros((self.n_samples, n_stacks, n_tiers), dtype=torch.float32, device=device)

        # ✅ PyTorch를 사용하여 컨테이너 순서를 무작위로 생성
        container_sequences = torch.rand((self.n_samples, n_containers), device=device).argsort(dim=-1).float() + 1

        # ✅ 스택을 랜덤하게 배정 (완전히 PyTorch 연산으로 변환)
        stack_fill_counts = torch.zeros((self.n_samples, n_stacks), dtype=torch.int32, device=device)

        for j in range(n_containers):
            valid_stacks = stack_fill_counts < n_tiers  # 공간이 있는 스택
            valid_stacks_float = valid_stacks.float()  # softmax를 위한 float 변환
            
            # ✅ GPU에서 직접 스택을 랜덤 선택
            stack_probs = valid_stacks_float / valid_stacks_float.sum(dim=-1, keepdim=True)  # 확률로 변환
            selected_stacks = torch.multinomial(stack_probs, 1).squeeze(dim=-1)  # 각 샘플별로 하나의 스택 선택

            tier_positions = stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks]
            data[torch.arange(self.n_samples, device=device), selected_stacks, tier_positions] = container_sequences[:, j]
            stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks] += 1

        # ✅ instance_type이 'upsidedown'이면 각 stack을 정렬 (완전 GPU 연산)
        if inst_type == 'upsidedown':
            mask = data > 0  # 0이 아닌 위치 찾기
            sorted_data, _ = torch.sort(torch.where(mask, data, torch.inf), dim=-1)  # 0을 무한대로 치환하여 정렬
            sorted_data[sorted_data == torch.inf] = 0  # 다시 0으로 복원
            data[:] = sorted_data  # 원본 데이터 업데이트

        print(f'{self.n_samples}개 data 생성시간: {round(time.time() - clock, 2)}초')

        batch_size, total_stacks, feature_dim = data.shape
        assert total_stacks == n_bays * n_rows
        data = data.reshape(batch_size, n_bays, n_rows, feature_dim)

        return data



    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index]




if __name__ == '__main__':

    layout = (35,2,4,6)
    inst_type = 'upsidedown'
    gen = Generator(
        load_data=None,
        seed=0,
        n_samples=1024,
        layout=layout,
        inst_type=inst_type,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ) 
    print(gen.data.shape)
    print(gen.data[0])

    file_name = f'./generator/eval_data/eval_data({layout[0]},{layout[1]},{layout[2]},{layout[3]})_{inst_type}.pt'
    
    torch.save(gen.data, file_name)

    data = torch.load(file_name)
    print(data.shape)
    print(data[0])