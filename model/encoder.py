import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    def forward(self, input):
        output= self.module(input)
        return input + output

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None
            return input

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.inf = 1e9
    def forward(self, Q, K, V, mask):
        d_k = self.d_k        
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k) 
        if mask is None:
            mask = torch.zeros_like(attn_score).bool()
        else:
            attn_score = attn_score.masked_fill(mask[:, None, None, :, 0].repeat(1, attn_score.size(1), 1, 1) == True, -self.inf)

        attn_dist = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_dist, V)

        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, is_encoder):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim//n_heads
        self.d_v = embed_dim//n_heads
        
        assert self.embed_dim % self.n_heads == 0

        self.is_encoder = is_encoder
        if self.is_encoder:
            self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, x, mask=None):
        Q,K,V = x
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1)
        if self.is_encoder:
            residual = Q
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
        
        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batchSize, seqLen_Q, -1)

        if self.is_encoder:
            output = residual + self.W_O(output)
        
        return output

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, ff_hidden = 64, normalization = 'instance', is_encoder=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.MHA = MultiHeadAttention(n_heads, embed_dim=embed_dim, is_encoder=is_encoder)
        self.BN1 = Normalization(embed_dim, normalization)
        self.BN2 = Normalization(embed_dim, normalization)
        
        self.FF_sub = SkipConnection(
                        nn.Sequential(
                            nn.Linear(embed_dim, ff_hidden),
                            nn.ReLU(),
                            nn.Linear(ff_hidden, embed_dim)
                        )
                    )
    def forward(self, x, mask=None):
        x = [x,x,x]
        x = self.BN1(self.MHA(x, mask=mask))
        x = self.BN2(self.FF_sub(x))
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.objective = 'workingtime'
        self.lstm = args.lstm
        if not self.lstm:
            if self.objective == 'workingtime':
                self.init_embed_dim  = 11
            elif self.objective == 'relocations':
                self.init_embed_dim  = 7
        else:
            if self.objective == 'workingtime':
                self.init_embed_dim  = 6
            elif self.objective == 'relocations':
                self.init_embed_dim  = 2
        self.empty_priority = None
        self.norm_priority = True
        self.norm_layout = True
        if self.norm_layout:
            assert self.objective == 'workingtime'
        self.add_fill_ratio = True
        self.add_layout_ratio = True
        self.add_travel_time = True
        if self.add_fill_ratio:
            self.init_embed_dim += 1
        if self.add_layout_ratio:
            self.init_embed_dim += 1
            assert self.objective == 'workingtime'
        if self.add_travel_time:
            self.init_embed_dim += 2
            assert self.objective == 'workingtime'
        self.embed_dim = args.embed_dim

        self.fcs = nn.Sequential(
            nn.Linear(self.init_embed_dim, self.embed_dim//2),
            nn.ReLU(),
            nn.Linear(self.embed_dim//2, self.embed_dim)
        ).to(self.device)
        if self.lstm:
            self.pos_enc = nn.Sequential(
                nn.Linear(1, 16, bias=True),
                nn.ReLU(),
                nn.Linear(16, 1, bias = True)
            ).to(self.device)
            self.init_stack_emb = nn.Sequential(
                nn.Linear(2, self.embed_dim//2, bias=True),
                nn.ReLU(),
                nn.Linear(self.embed_dim//2, self.embed_dim, bias=True)
            ).to(self.device)
            self.LSTM = nn.LSTM(
                input_size = self.embed_dim,
                hidden_size = self.embed_dim,
                batch_first = True,
                num_layers = 1
            ).to(self.device)
            self.LSTM_embed = nn.Linear(self.embed_dim*2, self.embed_dim, bias=True).to(self.device)
            self.fcs3 = nn.Sequential(
                nn.Linear(self.embed_dim * 2, args.ff_hidden),
                nn.ReLU(),
                nn.Linear(args.ff_hidden, self.embed_dim)
            ).to(self.device)

        self.encoder_layers = nn.ModuleList(
            [MultiHeadAttentionLayer(args.n_heads, self.embed_dim, args.ff_hidden)
            for _ in range(args.n_encode_layers)]
        ).to(self.device)

        self.bay_embedding = args.bay_embedding
        if self.bay_embedding:
            self.fcs2 = nn.Sequential(
                nn.Linear(self.embed_dim * 2, args.ff_hidden),
                nn.ReLU(),
                nn.Linear(args.ff_hidden, self.embed_dim)
            ).to(self.device)



    def expert_feature(self, x, tier, empty_prio, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows):
        len_mask = torch.where(x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2).to(self.device)
        change_empty = torch.where(x == 0., empty_prio, x).to(self.device)
        min_due = torch.min(change_empty, dim=2)[0].view(batch, stack, 1).to(self.device)

        #Top Due
        top_ind=stack_len-1
        top_ind=torch.where(top_ind>=0,top_ind,0).to(self.device)
        top_val=torch.gather(change_empty,dim=2,index=top_ind[:,:,None]).to(self.device)

        #Well-Located
        is_well = torch.where(min_due >= top_val, 1., 0.).to(self.device)
        #Is target
        is_target = torch.where((min_due == 1) & (stack_len.unsqueeze(-1) > 0), 1., 0.).to(self.device)
        is_target[torch.where(torch.sum(is_target, dim=1)==0)[0],0] +=1
        #Stack_Height
        stack_height = stack_len.view(batch,stack,1)

        # stack → row
        stack_indices = torch.arange(stack).to(self.device)  # 0, 1, 2, ..., num_stacks-1
        stack_rows = stack_indices % n_rows + 1
        stack_rows = stack_rows.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks)
        # find target stack
        target_mask = is_target.squeeze(-1).bool()  # shape: (batch, stack)
        target_rows = stack_rows[target_mask]  # target stack row

        # stack → bay
        stack_bays = stack_indices // n_rows + 1
        stack_bays = stack_bays.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks)
        target_bays = stack_bays[target_mask]

        # gap btw target stack and each stack
        row_diff = torch.abs(stack_rows - target_rows.unsqueeze(-1)).float().to(self.device)  # (batch, stack)
        row_diff = row_diff.unsqueeze(-1)  # (batch, stack, 1)
        bay_diff = torch.abs(stack_bays - target_bays.unsqueeze(-1)).float().to(self.device)  # (batch, stack)
        bay_diff = bay_diff.unsqueeze(-1)  # (batch, stack, 1)

        # Maximum Due Date
        md = torch.max(x,dim=2)[0].view(batch, stack, 1).to(self.device)

        if self.add_travel_time:
            reloc_time = t_row * row_diff + t_bay * bay_diff + t_acc * (bay_diff > 0)
            truck_time = t_row * stack_rows.unsqueeze(-1)
            travel_time = torch.cat([reloc_time, truck_time], dim=-1)
            max_val = travel_time.amax(dim=(1,2), keepdim=True)
            travel_time = travel_time / max_val

        if self.norm_priority:
            n_plus_one = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1
            min_due = min_due / n_plus_one
            top_val = top_val / n_plus_one
            md = md / n_plus_one

        if self.norm_layout:
            if n_rows > 1:
                stack_rows = (stack_rows - 1) / (n_rows - 1)
                row_diff = row_diff / (n_rows - 1)
            else:
                stack_rows = stack_rows - 1
            if n_bays > 1:
                stack_bays = (stack_bays - 1) / (n_bays - 1)
                bay_diff = bay_diff / (n_bays - 1)
            else:
                stack_bays = stack_bays - 1

        if self.objective == 'workingtime':
            ft = torch.cat([min_due, top_val, is_well, is_target, stack_height, tier - stack_height,
                            stack_rows.unsqueeze(-1), row_diff,
                            stack_bays.unsqueeze(-1), bay_diff,
                            md], dim=2).to(self.device)
        elif self.objective == 'relocations':
            ft = torch.cat([min_due, top_val, is_well, is_target, stack_height, tier - stack_height,
                            md], dim=2).to(self.device)

        if self.add_fill_ratio:
            nonzero_count = (x > 0).sum(dim=(1, 2), keepdim=True)
            layout_size = x.shape[1] * x.shape[2]
            fill_ratio = nonzero_count / layout_size
            fill_ratio = fill_ratio.expand(-1, x.shape[1], -1)
            ft = torch.cat([ft, fill_ratio], dim=2)

        if self.add_layout_ratio:
            layout_ratio = torch.full_like(min_due, n_bays/n_rows)
            ft = torch.cat([ft, layout_ratio], dim=2)

        if self.add_travel_time:
            ft = torch.cat([ft, travel_time], dim=2)
        
        ft = self.fcs(ft)

        return ft


    def stack_position_feature(self, x, tier, empty_prio, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows):
        len_mask = torch.where(x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2).to(self.device)
        change_empty = torch.where(x == 0., empty_prio, x).to(self.device)
        min_due = torch.min(change_empty, dim=2)[0].view(batch, stack, 1).to(self.device)

        #Is target
        is_target = torch.where((min_due == 1) & (stack_len.unsqueeze(-1) > 0), 1., 0.).to(self.device)
        is_target[torch.where(torch.sum(is_target, dim=1)==0)[0],0] +=1
        #Stack_Height
        stack_height = stack_len.view(batch,stack,1)

        # stack → row
        stack_indices = torch.arange(stack).to(self.device)  # 0, 1, 2, ..., num_stacks-1
        stack_rows = stack_indices % n_rows + 1
        stack_rows = stack_rows.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks)
        # find target stack
        target_mask = is_target.squeeze(-1).bool()  # shape: (batch, stack)
        target_rows = stack_rows[target_mask]  # target stack row

        # stack → bay
        stack_bays = stack_indices // n_rows + 1
        stack_bays = stack_bays.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks)
        target_bays = stack_bays[target_mask]

        # gap btw target stack and each stack
        row_diff = torch.abs(stack_rows - target_rows.unsqueeze(-1)).float().to(self.device)  # (batch, stack)
        row_diff = row_diff.unsqueeze(-1)  # (batch, stack, 1)
        bay_diff = torch.abs(stack_bays - target_bays.unsqueeze(-1)).float().to(self.device)  # (batch, stack)
        bay_diff = bay_diff.unsqueeze(-1)  # (batch, stack, 1)

        if self.add_travel_time:
            reloc_time = t_row * row_diff + t_bay * bay_diff + t_acc * (bay_diff > 0)
            truck_time = t_row * stack_rows.unsqueeze(-1)
            travel_time = torch.cat([reloc_time, truck_time], dim=-1)
            max_val = travel_time.amax(dim=(1,2), keepdim=True)
            travel_time = travel_time / max_val

        if self.norm_layout:
            if n_rows > 1:
                stack_rows = (stack_rows - 1) / (n_rows - 1)
                row_diff = row_diff / (n_rows - 1)
            else:
                stack_rows = stack_rows - 1
            if n_bays > 1:
                stack_bays = (stack_bays - 1) / (n_bays - 1)
                bay_diff = bay_diff / (n_bays - 1)
            else:
                stack_bays = stack_bays - 1

        if self.objective == 'workingtime':
            ft = torch.cat([stack_height, tier - stack_height,
                            stack_rows.unsqueeze(-1), row_diff,
                            stack_bays.unsqueeze(-1), bay_diff
                            ], dim=2).to(self.device)
        elif self.objective == 'relocations':
            ft = torch.cat([stack_height, tier - stack_height,
                            ], dim=2).to(self.device)

        if self.add_fill_ratio:
            nonzero_count = (x > 0).sum(dim=(1, 2), keepdim=True)
            layout_size = x.shape[1] * x.shape[2]
            fill_ratio = nonzero_count / layout_size
            fill_ratio = fill_ratio.expand(-1, x.shape[1], -1)
            ft = torch.cat([ft, fill_ratio], dim=2)

        if self.add_layout_ratio:
            layout_ratio = torch.full_like(min_due, n_bays/n_rows)
            ft = torch.cat([ft, layout_ratio], dim=2)

        if self.add_travel_time:
            ft = torch.cat([ft, travel_time], dim=2)
        
        ft = self.fcs(ft)

        return ft


    def forward(self, x, n_bays, n_rows, t_acc, t_bay, t_row, t_pd, mask=None):
        batch,stack,tier = x.size()

        if not self.lstm: # how to fill empty slot when not using lstm
            if self.empty_priority is None:
                empty_priority = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1
            else:
                assert stack*tier+1 < self.empty_priority
                empty_priority = torch.full((x.shape[0], 1, 1), self.empty_priority).to(self.device)

            x = self.expert_feature(x, tier, empty_priority, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows).to(self.device)

        else:
            empty_priority = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1 # change empty values for computational convenience

            # stack position features (row, bay)
            x2 = self.stack_position_feature(x, tier, empty_priority, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows).to(self.device)

            # lstm
            x = x.clone()
            x = torch.where(x > 0, 1 - (x - 1) / x.amax(dim=(1, 2), keepdim=True), x)
            x = x.view(batch, stack, tier, 1)

            asc = self.pos_enc(torch.linspace(0, 1, tier, device=self.device).repeat(batch, stack, 1).unsqueeze(-1)) # height info
            x = torch.cat([x, asc], dim=3)
            x = self.init_stack_emb(x)
            x = x.view(batch * stack, tier, self.embed_dim)
            output, (hidden_states, _) = self.LSTM(x) # output: each LSTM layer's output, hidden_states: last LSTM layer's output
            o = torch.mean(output, dim=1).view(batch, stack, self.embed_dim)
            h = hidden_states[0,:,:].view(batch, stack, self.embed_dim)
            x = torch.cat([o,h], dim=2)
            x = self.LSTM_embed(x)

            # position + lstm
            x = torch.cat([x, x2], dim=2)
            x = self.fcs3(x)

        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)

            if self.bay_embedding: # use bay-wise embedding
                x_reshaped = x.view(x.shape[0], n_bays, n_rows, x.shape[-1])
                bay_emb = x_reshaped.mean(dim=2)
                bay_emb = bay_emb.unsqueeze(2).repeat(1, 1, n_rows, 1).reshape(bay_emb.shape[0], -1, bay_emb.shape[-1])
                x = torch.cat([x, bay_emb], dim=2)
                x = self.fcs2(x)

        return (x, torch.mean(x, dim=1))

