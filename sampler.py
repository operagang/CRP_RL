import torch.nn as nn 
import torch
class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
        return; next_node: (batch, 1)
        TopKSampler <=> greedy; sample one with biggest probability
        CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
    """

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        return torch.topk(logits, self.n_samples, dim=1)[1]  # == torch.argmax(log_p, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)

class New_Sampler(Sampler):
    def __init__(self, T = 10 , **kwargs):
        super().__init__(**kwargs)
        self.T = T
    def forward(self, logits):
        #print(logits)
        new_logits = logits/self.T
        p_logits = torch.softmax(new_logits, dim=1)
        #print("new:", p_logits)
        #print("old:", torch.softmax(logits, dim=1))
        #print("------------------------------")
        #self.T = self.T * 0.95
        return torch.multinomial(p_logits, self.n_samples)
if __name__ == "__main__":
    sampler = New_Sampler()
    print(sampler(torch.tensor([[-7, -0.0001, -3, -12, -1e9], [-25, 55, -25, -25, -1e9]])))