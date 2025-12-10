import torch
import torch.nn as nn
import math
from model.encoder import Encoder, MultiHeadAttention
from env.env import Env
from model.sampler import TopKSampler, CategoricalSampler, New_Sampler

# from encoder_Attention import Encoder, MultiHeadAttention
# from Env_V4 import Env
# from sampler import TopKSampler, CategoricalSampler, New_Sampler
# from data_V4 import generate_data
# from decoder_utils import concat_embedding, concat_graph_embedding

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tanh_c = args.tanh_c
        self.samplers = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}
        self.sampler = None

        self.encoder = Encoder(args).to(self.device)
        self.W_target = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_global = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_K1 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_K2 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_Q = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.W_V = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.MHA = MultiHeadAttention(args.n_heads, args.embed_dim, is_encoder=False)

        # """"""
        # self.bay_diff = []
        # self.row_diff = []
        # self.well_located = []

        # mask others with a learnable token
        # self.visible_k   = getattr(args, 'visible_k', 20)
        # init_mask_token = float(self.visible_k + 1)
        # self.mask_token = nn.Parameter(torch.tensor(init_mask_token, device=self.device))

    
    # """"""
    # def save_log(self, actions, env):
    #     dest_idxs = actions.squeeze()
    #     n_bays = env.n_bays
    #     n_rows = env.n_rows
    #     for i in range(env.empty.shape[0]):
    #         if not env.empty[i]:
    #             s_bay = env.target_stack[i] // n_rows + 1
    #             s_row = env.target_stack[i] % n_rows + 1
    #             d_bay = dest_idxs[i] // n_rows + 1
    #             d_row = dest_idxs[i] % n_rows + 1
    #             self.bay_diff.append(abs(s_bay-d_bay).item())
    #             self.row_diff.append(abs(s_row-d_row).item())

    #             top = env.x[i][env.target_stack[i]][(env.x[i][env.target_stack[i]] != 0).nonzero(as_tuple=True)[0][-1]]
    #             d_stack = env.x[i][dest_idxs[i]].clone()
    #             d_stack[d_stack == 0] = 100000
    #             self.well_located.append((top < d_stack.min()).item())




    def set_sampler(self, decode_type):
        self.sampler = self.samplers[decode_type]


    def forward(self, x, max_retrievals):

        batch, n_bays, n_rows, max_tiers = x.size()
        max_stacks = n_bays * n_rows

        cost = torch.zeros(batch).to(self.device) # mini_batch * pomo_size 수
        ll = torch.zeros(batch).to(self.device) # mini_batch * pomo_size 수

        env = Env(self.device, x, max_retrievals) # max_retrievals 의미 X

        cost = cost + env.clear()

        """ 1. encoder """
        encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
        """"""""""""""""""

        """ 2. encoder in online setting """
        # x_new = env.x.clone()
        # batch_max = x_new.view(x_new.shape[0], -1).amax(dim=1)
        # mask = x_new > 20  # shape: [5, 16, 6]
        # for b in range(x_new.shape[0]):
        #     x_new[b][mask[b]] = batch_max[b]
        #     # x_new[b][mask[b]] = 21
        # encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

        # x_new = x.clone()
        # mask = x_new > self.visible_k
        # x_new[mask] = self.mask_token
        # encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
        """"""""""""""""""""""""""""""""""""

        node_embeddings, graph_embedding = encoder_output
        target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :] # target stack의 embedding
        mask = env.create_mask()

        for i in range(max_stacks * max_tiers * max_tiers): # upper bound of relocations
            assert i < max_stacks * max_tiers * max_tiers - 1, "끝까지 empty 되지 않음"

            # decoder
            # context vector는 target embedding과 global embedding으로 구성
            context = (self.W_target(target_embeddings) + self.W_global(graph_embedding)).unsqueeze(1)
            node_keys = self.W_K1(node_embeddings)
            node_values = self.W_V(node_embeddings)
            query_ = self.W_Q(self.MHA([context, node_keys, node_values])) # MHA를 통한 context vector 업데이트
            key_ = self.W_K2(node_embeddings)

            logits = torch.matmul(query_, key_.permute(0, 2, 1)).squeeze(1) / math.sqrt(query_.size(-1))
            logits = self.tanh_c * torch.tanh(logits)
            logits = logits - mask.squeeze(-1) * 1e9
            log_p = torch.log_softmax(logits, dim=1) # action 선택 확률의 log값

            actions = self.sampler(log_p) # action 선택

            # """"""
            # self.save_log(actions, env)

            tmp_log_p = log_p.clone()
            tmp_log_p[(env.empty | env.early_stopped), :] = 0 # 이미 종료되었던 문제는 확률값 0으로 (반드시 step 이전에 실행)
            ll = ll + torch.gather(input=tmp_log_p, dim=1, index=actions).squeeze(-1).to(self.device) # log likelihood
            
            cost = cost + env.step(dest_index=actions)

            if env.all_terminated():
                break

            """ 1. encoder """
            encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
            """"""""""""""""""

            """ 2. encoder in online setting """
            # x_new = env.x.clone()
            # batch_max = x_new.view(x_new.shape[0], -1).amax(dim=1)
            # mask = x_new > 20  # shape: [5, 16, 6]
            # for b in range(x_new.shape[0]):
            #     x_new[b][mask[b]] = batch_max[b]
            #     # x_new[b][mask[b]] = 21
            # encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

            # x_new = x.clone()
            # mask = x_new > self.visible_k
            # x_new[mask] = self.mask_token
            # encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
            """"""""""""""""""""""""""""""""""""

            node_embeddings, graph_embedding = encoder_output
            target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]
            mask = env.create_mask()

        return cost, ll, env.relocations, env.wt_lb










