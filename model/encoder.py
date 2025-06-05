import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SkipConnection(nn.Module):
    """ SkipConnection 구현 
        (Q,K,V,mask) -> (Q+output), attn_dist
    """
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    def forward(self, input):
        output= self.module(input)
        return input + output

class Normalization(nn.Module): #wounterkool git 참고
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class ScaledDotProductAttention(nn.Module): #Tsp_Attention.ipynb
    """ Attention(Q,K,V) = softmax(QK^T/root(d_k))V
    """
    def __init__(self, d_k):# d_k: head를 나눈 이후의 key dimension
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k # key dimenstion
        self.inf = 1e9
        #self.dropout = nn.Dropout(0.5)
    def forward(self, Q, K, V, mask):
        # key, query의 곱을 통해 attention weight를 계산하고 value의 weighted sum인 output을 생성
        # input: Q, K, V, mask (query, key, value, padding 및 시간을 반영하기 위한 mask)
        # output: output, attn_dist (value의 weight sum, attention weight)
        # dim of Q,K,V: batchSize x n_heads x seqLen x d_k(d_v)
        d_k = self.d_k        
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k) 
                    # dim of attn_score: batchSize x n_heads x seqLen_Q x seqLen_K
                    #wj) batch matrix multiplication
        if mask is None:
            mask = torch.zeros_like(attn_score).bool()
        else:
            attn_score = attn_score.masked_fill(mask[:, None, None, :, 0].repeat(1, attn_score.size(1), 1, 1) == True, -self.inf)

        attn_dist = F.softmax(attn_score, dim=-1)  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x n_heads x seqLen x d_v

        return output, attn_dist

class SingleHeadAttention(nn.Module):
    def __init__(self, clip=10, head_depth=16, inf=1e+10, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.inf = inf
        self.scale = math.sqrt(head_depth)

    # self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        """ Q: (batch, n_heads, q_seq(=max_stacks or =1), head_depth)
            K: (batch, n_heads, k_seq(=max_stacks), head_depth)
            logits: (batch, n_heads, q_seq(this could be 1), k_seq)
            mask: (batch, max_stacks, 1), e.g. tf.Tensor([[ True], [ True], [False]])
            mask[:,None,None,:,0]: (batch, 1, 1, stacks) ==> broadcast depending on logits shape
            [True] -> [1 * -np.inf], [False] -> [logits]
            K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
        """
        Q, K, V = x
        logits = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        logits = self.clip * torch.tanh(logits)

        if mask is not None:
            return logits.masked_fill(mask.permute(0, 2, 1) == True, -self.inf)
        return logits

class MultiHeadAttention(nn.Module):
    """ Skip_Connection 은 Built_in 되어있습니다.
        Norm은 따로 진행합니다. (tsp_attention.ipynb와 다름)
    """
    def __init__(self, n_heads, embed_dim, is_encoder):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim//n_heads
        self.d_v = embed_dim//n_heads
        
        assert self.embed_dim % self.n_heads == 0 #embed_dim = n_heads * head_depth

        self.is_encoder = is_encoder
        if self.is_encoder:
            self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
            # self.layerNorm = nn.LayerNorm(embed_dim, 1e-6) # layer normalization
        self.attention = ScaledDotProductAttention(self.d_k)
        #self.init_parameters()
    def init_parameters(self):
        #JK: 필요하다면 구현해야함
        pass
    def forward(self, x, mask=None):
        Q,K,V = x
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1) # decoder의 경우 query와 key의 length가 다를 수 있음
        # Query, Key, Value를 (n_heads)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # dim : batchSize x seqLen x embed_dim -> batchSize x seqLen x n_heads x d_k
        if self.is_encoder:
            residual = Q
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
        
        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # dim : batchSize x seqLen x n_heads x d_k -> batchSize x n_heads x seqLen x d_k
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous()  # dim : batchSize x n_heads x seqLen x d_k -> batchSize x seqLen x n_heads x d_k
        output = output.view(batchSize, seqLen_Q, -1)  # dim : batchSize x seqLen x n_heads x d_k -> batchSize x seqLen x d_model

        # Linear Projection, Residual sum
        if self.is_encoder:
            output = residual + self.W_O(output)
        
        return output

class MultiHeadAttentionLayer(nn.Module): #Self-Attention
    """ h_ = BN(h+MHA(h))
        h = BN(h_ + FF(h_))
    """
    def __init__(self, n_heads, embed_dim, ff_hidden = 64, normalization = 'instance', is_encoder=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.MHA = MultiHeadAttention(n_heads, embed_dim=embed_dim, is_encoder=is_encoder) #Maybe Modified
        self.BN1 = Normalization(embed_dim, normalization)
        self.BN2 = Normalization(embed_dim, normalization)
        
        self.FF_sub = SkipConnection(
                        nn.Sequential(
                            nn.Linear(embed_dim, ff_hidden), #bias = True by default
                            nn.ReLU(),
                            nn.Linear(ff_hidden, embed_dim)  #bias = True by default
                        )
                    )
    def forward(self, x, mask=None):
        #######################################
        #With BatchNorm/InstanceNorm
        x = [x,x,x] # Self_Attention
        x = self.BN1(self.MHA(x, mask=mask))
        x = self.BN2(self.FF_sub(x))
        #######################################
        #######################################
        #Without BatchNorm
        #x = [x,x,x]
        #x = self.FF_sub(self.MHA(x, mask=mask))    
        #######################################    
        return x
    
"""
class MultiHeadAttentionLayer(nn.Module): #Self-Attention
    
    def __init__(self, n_heads, embed_dim, ff_hidden, normalization = 'instance', is_encoder=True, init_resweight = 0, resweight_trainable=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)
        self.MHA = MultiHeadAttention(n_heads, embed_dim=embed_dim, is_encoder=is_encoder) #Maybe Modified
        
        
        self.FF_sub = nn.Sequential(
                            nn.Linear(embed_dim, ff_hidden), #bias = True by default
                            nn.ReLU(),
                            nn.Linear(ff_hidden, embed_dim)  #bias = True by default
                        )

    def forward(self, x, mask=None):
        #######################################
        #With BatchNorm/InstanceNorm
        #x = [x,x,x] # Self_Attention
        #x = self.N1(self.MHA(x, mask=mask))
        #x = self.N2(self.FF_sub(x))
        #######################################
        #######################################
        #ReZero
        t = [x,x,x]
        x = x + self.resweight * self.MHA(t, mask=mask)
        x = x + self.resweight * self.FF_sub(x)
        #######################################    
        return x
"""
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.objective = args.objective
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
        self.empty_priority = args.empty_priority
        self.norm_priority = args.norm_priority
        self.norm_layout = args.norm_layout
        if self.norm_layout:
            assert self.objective == 'workingtime'
        self.add_fill_ratio = args.add_fill_ratio
        self.add_layout_ratio = args.add_layout_ratio
        self.add_travel_time = args.add_travel_time
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
            nn.Linear(self.init_embed_dim, self.embed_dim//2), #bias = True by default
            nn.ReLU(),
            nn.Linear(self.embed_dim//2, self.embed_dim)
        ).to(self.device)
        if self.lstm:
            self.pos_enc = nn.Sequential(
                nn.Linear(1, 16, bias=True),
                nn.ReLU(),
                #nn.Dropout(.5),
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
                nn.Linear(self.embed_dim * 2, args.ff_hidden), #bias = True by default
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
                nn.Linear(self.embed_dim * 2, args.ff_hidden), #bias = True by default
                nn.ReLU(),
                nn.Linear(args.ff_hidden, self.embed_dim)
            ).to(self.device)



    def en(self, x, tier, empty_prio, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows):
        len_mask = torch.where(x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2).to(self.device)
        # total_len = torch.sum(stack_len+1, dim=1).view(batch, 1, 1).repeat(1, stack, tier).to(torch.float32) + 1
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
        is_target[torch.where(torch.sum(is_target, dim=1)==0)[0],0] +=1 #All_Empty인 곳비워져있는곳은 첫번째 스택을target으로 지정
        #Stack_Height
        stack_height = stack_len.view(batch,stack,1)

        # ✅ stack → row 변환
        stack_indices = torch.arange(stack).to(self.device)  # 0, 1, 2, ..., num_stacks-1
        stack_rows = stack_indices % n_rows + 1  # stack을 row로 변환
        stack_rows = stack_rows.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks) 형태로 확장
        # ✅ target stack 찾기
        target_mask = is_target.squeeze(-1).bool()  # shape: (batch, stack)
        target_rows = stack_rows[target_mask]  # target stack의 row

        # ✅ stack → bay 변환
        stack_bays = stack_indices // n_rows + 1  # stack을 row로 변환
        stack_bays = stack_bays.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks) 형태로 확장
        target_bays = stack_bays[target_mask]

        # ✅ target stack의 row와 각 stack의 row 차이 계산
        row_diff = torch.abs(stack_rows - target_rows.unsqueeze(-1)).float().to(self.device)  # (batch, stack) 크기
        # ✅ min_due와 동일한 shape으로 변환
        row_diff = row_diff.unsqueeze(-1)  # (batch, stack, 1) 크기로 맞춤
        bay_diff = torch.abs(stack_bays - target_bays.unsqueeze(-1)).float().to(self.device)  # (batch, stack) 크기
        bay_diff = bay_diff.unsqueeze(-1)  # (batch, stack, 1) 크기로 맞춤

        #Maximum Due Date
        md = torch.max(x,dim=2)[0].view(batch, stack, 1).to(self.device)
        # md = torch.where(md == 0., empty_prio, md).to(self.device)

        if self.add_travel_time:
            reloc_time = t_row * row_diff + t_bay * bay_diff + t_acc * (bay_diff > 0)
            truck_time = t_row * stack_rows.unsqueeze(-1) # (1024, 8) → (1024, 8, 1)로 변환
            travel_time = torch.cat([reloc_time, truck_time], dim=-1)
            max_val = travel_time.amax(dim=(1,2), keepdim=True)
            travel_time = travel_time / max_val


        if self.norm_priority:
            n_plus_one = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1
            min_due = min_due / n_plus_one
            top_val = top_val / n_plus_one
            md = md / n_plus_one


        # n_bays = x.shape[1] // n_rows
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


    def en2(self, x, tier, empty_prio, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows):
        len_mask = torch.where(x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2).to(self.device)
        # total_len = torch.sum(stack_len+1, dim=1).view(batch, 1, 1).repeat(1, stack, tier).to(torch.float32) + 1
        change_empty = torch.where(x == 0., empty_prio, x).to(self.device)
        min_due = torch.min(change_empty, dim=2)[0].view(batch, stack, 1).to(self.device)

        #Is target
        is_target = torch.where((min_due == 1) & (stack_len.unsqueeze(-1) > 0), 1., 0.).to(self.device)
        is_target[torch.where(torch.sum(is_target, dim=1)==0)[0],0] +=1 #All_Empty인 곳비워져있는곳은 첫번째 스택을target으로 지정
        #Stack_Height
        stack_height = stack_len.view(batch,stack,1)

        # ✅ stack → row 변환
        stack_indices = torch.arange(stack).to(self.device)  # 0, 1, 2, ..., num_stacks-1
        stack_rows = stack_indices % n_rows + 1  # stack을 row로 변환
        stack_rows = stack_rows.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks) 형태로 확장
        # ✅ target stack 찾기
        target_mask = is_target.squeeze(-1).bool()  # shape: (batch, stack)
        target_rows = stack_rows[target_mask]  # target stack의 row

        # ✅ stack → bay 변환
        stack_bays = stack_indices // n_rows + 1  # stack을 row로 변환
        stack_bays = stack_bays.unsqueeze(0).expand(batch, -1).to(self.device)  # (batch, num_stacks) 형태로 확장
        target_bays = stack_bays[target_mask]

        # ✅ target stack의 row와 각 stack의 row 차이 계산
        row_diff = torch.abs(stack_rows - target_rows.unsqueeze(-1)).float().to(self.device)  # (batch, stack) 크기
        # ✅ min_due와 동일한 shape으로 변환
        row_diff = row_diff.unsqueeze(-1)  # (batch, stack, 1) 크기로 맞춤
        bay_diff = torch.abs(stack_bays - target_bays.unsqueeze(-1)).float().to(self.device)  # (batch, stack) 크기
        bay_diff = bay_diff.unsqueeze(-1)  # (batch, stack, 1) 크기로 맞춤

        if self.add_travel_time:
            reloc_time = t_row * row_diff + t_bay * bay_diff + t_acc * (bay_diff > 0)
            truck_time = t_row * stack_rows.unsqueeze(-1) # (1024, 8) → (1024, 8, 1)로 변환
            travel_time = torch.cat([reloc_time, truck_time], dim=-1)
            max_val = travel_time.amax(dim=(1,2), keepdim=True)
            travel_time = travel_time / max_val

        # n_bays = x.shape[1] // n_rows
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

        if not self.lstm:
            if self.empty_priority is None:
                empty_priority = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1
            else:
                assert stack*tier+1 < self.empty_priority, "empty_priority 보다 컨테이너 수가 더 많을 수 있음"
                empty_priority = torch.full((x.shape[0], 1, 1), self.empty_priority).to(self.device)

            x = self.en(x, tier, empty_priority, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows).to(self.device)

        else:
            empty_priority = torch.sum(x > 0, dim=[1,2]).view(x.shape[0], 1, 1) + 1
            x2 = self.en2(x, tier, empty_priority, t_acc, t_bay, t_row, t_pd, batch, stack, n_bays, n_rows).to(self.device)

            x = x.clone()
            x = torch.where(x > 0, 1 - (x - 1) / x.amax(dim=(1, 2), keepdim=True), x)
            x = x.view(batch, stack, tier, 1)

            asc = self.pos_enc(torch.linspace(0, 1, tier, device=self.device).repeat(batch, stack, 1).unsqueeze(-1))
            # desc = self.pos_enc2(torch.linspace(1, 0, tier, device=self.device).repeat(batch, stack, 1).unsqueeze(-1))

            # x = torch.cat([x, asc, desc], dim=3)
            x = torch.cat([x, asc], dim=3)
            x = self.init_stack_emb(x)
            x = x.view(batch * stack, tier, self.embed_dim)
            output, (hidden_states, _) = self.LSTM(x)
            o = torch.mean(output, dim=1).view(batch, stack, self.embed_dim)
            h = hidden_states[0,:,:].view(batch, stack, self.embed_dim)
            x = torch.cat([o,h], dim=2)
            x = self.LSTM_embed(x)

            x = torch.cat([x, x2], dim=2)
            x = self.fcs3(x)

            pass




        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)

            if self.bay_embedding:
                x_reshaped = x.view(x.shape[0], n_bays, n_rows, x.shape[-1])
                bay_emb = x_reshaped.mean(dim=2)
                bay_emb = bay_emb.unsqueeze(2).repeat(1, 1, n_rows, 1).reshape(bay_emb.shape[0], -1, bay_emb.shape[-1])
                x = torch.cat([x, bay_emb], dim=2)
                x = self.fcs2(x)


        return (x, torch.mean(x, dim=1))

