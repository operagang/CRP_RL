import torch
import torch.nn as nn
import math
from model.encoder import Encoder, MultiHeadAttention
from env.env import Env
from model.sampler import TopKSampler, CategoricalSampler, New_Sampler


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

        self.online = args.online
        if self.online:
            self.online_known_num  = args.online_known_num
            init_mask_token = float(self.online_known_num + 1)
            self.mask_token = nn.Parameter(torch.tensor(init_mask_token, device=self.device))
  



    def set_sampler(self, decode_type):
        self.sampler = self.samplers[decode_type]


    def forward(self, x, max_retrievals):

        batch, n_bays, n_rows, max_tiers = x.size()
        max_stacks = n_bays * n_rows

        cost = torch.zeros(batch).to(self.device) # mini_batch * pomo_size
        ll = torch.zeros(batch).to(self.device) # mini_batch * pomo_size

        env = Env(self.device, x, max_retrievals) # max_retrivevals: early stopping parameter (not used)

        cost = cost + env.clear()

        if not self.online:
            encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
        else:
            x_new = env.x.clone()
            mask = x_new > self.online_known_num
            x_new[mask] = self.mask_token
            encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

        node_embeddings, graph_embedding = encoder_output
        target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :] # target stack embedding
        mask = env.create_mask()

        for i in range(max_stacks * max_tiers * max_tiers): # upper bound of relocations
            assert i < max_stacks * max_tiers * max_tiers - 1

            # decoder
            # context vector consists of target embedding and global embedding
            context = (self.W_target(target_embeddings) + self.W_global(graph_embedding)).unsqueeze(1)
            node_keys = self.W_K1(node_embeddings)
            node_values = self.W_V(node_embeddings)
            query_ = self.W_Q(self.MHA([context, node_keys, node_values]))
            key_ = self.W_K2(node_embeddings)

            logits = torch.matmul(query_, key_.permute(0, 2, 1)).squeeze(1) / math.sqrt(query_.size(-1))
            logits = self.tanh_c * torch.tanh(logits)
            logits = logits - mask.squeeze(-1) * 1e9
            log_p = torch.log_softmax(logits, dim=1)

            actions = self.sampler(log_p)

            tmp_log_p = log_p.clone()
            tmp_log_p[(env.empty | env.early_stopped), :] = 0 # set prob value as 0 for previously terminated instances (excute bfr calling step fn)
            ll = ll + torch.gather(input=tmp_log_p, dim=1, index=actions).squeeze(-1).to(self.device)
            
            cost = cost + env.step(dest_index=actions)

            if env.all_terminated():
                break

            if not self.online:
                encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
            else:
                x_new = env.x.clone()
                mask = x_new > self.online_known_num
                x_new[mask] = self.mask_token
                encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

            node_embeddings, graph_embedding = encoder_output
            target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]
            mask = env.create_mask()

        return cost, ll










