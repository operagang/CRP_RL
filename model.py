import torch
import torch.nn as nn
from decoder import Decoder


class Model(nn.Module):

    def __init__(self, device, embed_dim, n_encode_layers, n_heads, ff_hidden, tanh_c):
        super().__init__()

        self.device = device
        self.decoder = Decoder(device=device, embed_dim=embed_dim, n_encode_layers=n_encode_layers,
                               n_heads=n_heads, ff_hidden=ff_hidden, tanh_c=tanh_c).to(self.device)


    # cost length ll:log_softmax  sum of probability  pi:predicted tour
    def forward(self, x, n_bays, n_rows):
        decoder_output = self.decoder(x, n_bays, n_rows)
        cost, ll, reloc = decoder_output
        return cost, ll, reloc






if __name__ == '__main__':
    from benchmarks import find_and_process_file

    # Example usage
    folder_path = "./Lee_instances"  # Replace with the folder containing your files
    inst_type   = "random"
    n_bays      = 2
    n_rows      = 16
    n_tiers     = 6

    id = 1
    container_tensor1, _ = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)
    print(container_tensor1.shape)  # Should be (1, n_bays * n_stacks, n_tiers)
    # print(container_tensor1)

    id = 2
    container_tensor2, _ = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)
    print(container_tensor2.shape)  # Should be (1, n_bays * n_stacks, n_tiers)
    # print(container_tensor2)

    
    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    embed_dim       = 128
    n_encode_layers = 3
    n_heads         = 8
    ff_hidden       = 512
    tanh_c          = 10

    inputs = torch.cat((container_tensor1, container_tensor2), dim=0).to(device)
    print(inputs.shape)


    ###
    inputs = torch.Tensor(
        [
            [[1,2,3],
             [4,5,0],
             [0,0,0]],
             [[2,3,0],
             [4,1,0],
             [5,0,0]]
        ]
    ).to(device)
    n_bays = 1; n_rows = 3
    ###


    model = Model(device=device, embed_dim=embed_dim, n_encode_layers=n_encode_layers, n_heads=n_heads, ff_hidden=ff_hidden, tanh_c=tanh_c)
    model = model.to(device)
    model.train()

    model(inputs, n_bays, n_rows)
    pass
