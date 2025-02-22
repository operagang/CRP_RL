import time
import torch
import numpy as np
from torch.utils.data import Dataset



class Generator(Dataset):
    def __init__(self, args, seed=None, eval=False, load_data=None):
        if not eval:
            self.n_samples = args.batch_size * args.batch_num
            self.seed = seed
        else:
            self.n_samples = args.eval_batch_size * args.eval_batch_num
            self.seed = args.eval_seed
        self.n_stacks = args.n_bays * args.n_rows  # 전체 stack 개수
        self.n_tiers = args.n_tiers
        self.n_containers = args.n_containers
        self.instance_type = args.instance_type
        self.n_bays = args.n_bays
        self.n_rows = args.n_rows
        self.device = args.device

        # ✅ 데이터 생성
        if load_data is None:
            self.data = self.generate_data_vectorized()
        else:
            self.data = torch.load(load_data)

    # def generate_data_vectorized(self):
    #     """
    #     NumPy 벡터 연산을 활용하여 빠르게 컨테이너 배치를 생성
    #     """
    #     if self.seed is not None:
    #         np.random.seed(self.seed)
    #         torch.manual_seed(self.seed)

    #     clock = time.time()
    #     data = np.zeros((self.n_samples, self.n_stacks, self.n_tiers), dtype=np.float32)

    #     # ✅ 컨테이너 순서를 한 번에 벡터 연산으로 생성
    #     container_sequences = np.tile(np.arange(1, self.n_containers + 1), (self.n_samples, 1))
    #     np.apply_along_axis(np.random.shuffle, 1, container_sequences)

    #     # ✅ 스택을 랜덤하게 배정 (벡터 연산 활용)
    #     stack_fill_counts = np.zeros((self.n_samples, self.n_stacks), dtype=int)
    #     for j in range(self.n_containers):
    #         valid_stacks = stack_fill_counts < self.n_tiers  # 공간이 있는 스택
    #         selected_stacks = np.array([
    #             np.random.choice(np.where(valid_stacks[i])[0]) for i in range(self.n_samples)
    #         ])

    #         tier_positions = stack_fill_counts[np.arange(self.n_samples), selected_stacks]
    #         data[np.arange(self.n_samples), selected_stacks, tier_positions] = container_sequences[:, j]
    #         stack_fill_counts[np.arange(self.n_samples), selected_stacks] += 1

    #     # ✅ instance_type이 'upsidedown'이면 각 stack을 정렬
    #     if self.instance_type == 'upsidedown':
    #         mask = data > 0  # 0이 아닌 위치 찾기
    #         sorted_data = np.sort(np.where(mask, data, np.inf), axis=-1)  # 0이 아닌 값 정렬, 0은 유지
    #         sorted_data[sorted_data == np.inf] = 0  # 다시 0으로 복원
    #         data[:] = sorted_data  # 원본 데이터 업데이트

    #     # ✅ PyTorch Tensor로 변환
    #     data = torch.tensor(data, dtype=torch.float32)
    #     print(f'{self.n_samples}개 data 생성시간 (벡터화): {round(time.time() - clock, 2)}초')

    #     return data

    def generate_data_vectorized(self):
        """
        완전히 PyTorch 기반으로 GPU 연산을 수행하여 컨테이너 배치를 생성
        """
        device = self.device  # ✅ GPU 사용 여부 결정
        if self.seed is not None:
            torch.manual_seed(self.seed)  # ✅ PyTorch 난수 시드 고정

        clock = time.time()

        # ✅ PyTorch Tensor를 사용하여 초기화 (GPU로 이동 가능)
        data = torch.zeros((self.n_samples, self.n_stacks, self.n_tiers), dtype=torch.float32, device=device)

        # ✅ PyTorch를 사용하여 컨테이너 순서를 무작위로 생성
        container_sequences = torch.rand((self.n_samples, self.n_containers), device=device).argsort(dim=-1).float() + 1

        # ✅ 스택을 랜덤하게 배정 (완전히 PyTorch 연산으로 변환)
        stack_fill_counts = torch.zeros((self.n_samples, self.n_stacks), dtype=torch.int32, device=device)

        for j in range(self.n_containers):
            valid_stacks = stack_fill_counts < self.n_tiers  # 공간이 있는 스택
            valid_stacks_float = valid_stacks.float()  # softmax를 위한 float 변환
            
            # ✅ GPU에서 직접 스택을 랜덤 선택
            stack_probs = valid_stacks_float / valid_stacks_float.sum(dim=-1, keepdim=True)  # 확률로 변환
            selected_stacks = torch.multinomial(stack_probs, 1).squeeze(dim=-1)  # 각 샘플별로 하나의 스택 선택

            tier_positions = stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks]
            data[torch.arange(self.n_samples, device=device), selected_stacks, tier_positions] = container_sequences[:, j]
            stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks] += 1

        # ✅ instance_type이 'upsidedown'이면 각 stack을 정렬 (완전 GPU 연산)
        if self.instance_type == 'upsidedown':
            mask = data > 0  # 0이 아닌 위치 찾기
            sorted_data, _ = torch.sort(torch.where(mask, data, torch.inf), dim=-1)  # 0을 무한대로 치환하여 정렬
            sorted_data[sorted_data == torch.inf] = 0  # 다시 0으로 복원
            data[:] = sorted_data  # 원본 데이터 업데이트

        print(f'{self.n_samples}개 data 생성시간: {round(time.time() - clock, 2)}초')

        return data



    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index]




if __name__ == '__main__':
    import argparse
    from datetime import datetime

    args = argparse.Namespace(
        lr = 0.000001,
        epochs = 1500,
        batch_size = 512*2, # 256
        batch_num = 100, # 20
        eval_batch_size = 512*2, # 256
        eval_batch_num = 1, # 5
        eval_seed = 0,
        embed_dim = 128,
        n_encode_layers = 3,
        n_heads = 8,
        ff_hidden = 512,
        tanh_c = 10,
        n_bays = 2,
        n_rows = 4,
        n_tiers = 6,
        instance_type = 'upsidedown',
        objective = 'workingtime', # or relocations
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        log_path = f"./train/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    gen = Generator(args)
    print(gen.data[0])

    a = torch.load('./train/20250220_103119/eval_data.pt')
    print(a[0])