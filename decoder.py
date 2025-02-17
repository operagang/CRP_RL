import torch
import torch.nn as nn
import math
from encoder import Encoder, MultiHeadAttention
from env import Env
from sampler import TopKSampler, CategoricalSampler, New_Sampler

# from encoder_Attention import Encoder, MultiHeadAttention
# from Env_V4 import Env
# from sampler import TopKSampler, CategoricalSampler, New_Sampler
# from data_V4 import generate_data
# from decoder_utils import concat_embedding, concat_graph_embedding

class Decoder(nn.Module):
    def __init__(self, 
                 device, 
                 embed_dim, 
                 n_encode_layers, 
                 n_heads,
                 ff_hidden,
                 tanh_c):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.tanh_c = tanh_c
        
        self.encoder = Encoder(device=device, embed_dim=embed_dim, n_layers=n_encode_layers, ff_hidden=ff_hidden, n_heads=n_heads).to(self.device)
        self.W_target = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_global = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.MHA = MultiHeadAttention(n_heads, embed_dim, is_encoder=False)



    def forward(self, x, n_bays, n_rows, decode_type='sampling'):
        sampler = {'greedy': TopKSampler(), 'sampling': CategoricalSampler(), 'new_sampling': New_Sampler()}.get(decode_type, None)

        batch, max_stacks, max_tiers = x.size()
        
        cost = torch.zeros(batch).to(self.device)
        ll = torch.zeros(batch).to(self.device)

        env = Env(self.device, x, n_bays, n_rows)

        cost += env.clear()

        # encoder
        encoder_output = self.encoder(env.x, n_rows)
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
            logits -= mask.squeeze(-1) * 1e9
            log_p = torch.log_softmax(logits, dim=1)

            if decode_type == 'new_sampling':
                actions = sampler(logits)
            else:
                actions = sampler(log_p)

            log_p[env.empty, :] = 0 # 반드시 step 이전에
            ll += torch.gather(input=log_p, dim=1, index=actions).squeeze(-1).to(self.device)

            cost += env.step(dest_index=actions)

            if env.all_empty():
                break

            # encoder
            encoder_output = self.encoder(env.x, n_rows)
            node_embeddings, graph_embedding = encoder_output
            target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)), env.target_stack, :]
            mask = env.create_mask()

        return cost, ll, env.relocations










