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


    def set_sampler(self, decode_type):
        self.sampler = self.samplers[decode_type]


    def forward(self, x, max_retrievals):

        batch, n_bays, n_rows, max_tiers = x.size()
        max_stacks = n_bays * n_rows

        cost = torch.zeros(batch).to(self.device)
        ll = torch.zeros(batch).to(self.device)

        env = Env(self.device, x, max_retrievals)

        cost = cost + env.clear()

        """ encoder """
        # encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

        x_new = env.x.clone()
        batch_max = x_new.view(x_new.shape[0], -1).amax(dim=1)
        mask = x_new > 20  # shape: [5, 16, 6]
        for b in range(x_new.shape[0]):
            x_new[b][mask[b]] = batch_max[b]
            # x_new[b][mask[b]] = 21
        encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
        """"""

        node_embeddings, graph_embedding = encoder_output
        target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]
        mask = env.create_mask()

        for i in range(max_stacks * max_tiers * max_tiers): # upper bound of relocations
            assert i < max_stacks * max_tiers * max_tiers - 1, "끝까지 empty 되지 않음"

            # decoder
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
            tmp_log_p[(env.empty | env.early_stopped), :] = 0 # 반드시 step 이전에
            ll = ll + torch.gather(input=tmp_log_p, dim=1, index=actions).squeeze(-1).to(self.device)
            
            cost = cost + env.step(dest_index=actions)

            if env.all_terminated():
                break

            """ encoder """
            # encoder_output = self.encoder(env.x, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)

            x_new = env.x.clone()
            batch_max = x_new.view(x_new.shape[0], -1).amax(dim=1)
            mask = x_new > 20  # shape: [5, 16, 6]
            for b in range(x_new.shape[0]):
                x_new[b][mask[b]] = batch_max[b]
                # x_new[b][mask[b]] = 21
            encoder_output = self.encoder(x_new, n_bays, n_rows, env.t_acc, env.t_bay, env.t_row, env.t_pd)
            """"""

            node_embeddings, graph_embedding = encoder_output
            target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]
            mask = env.create_mask()

        return cost, ll, env.relocations, env.wt_lb










