import torch.optim
import torch.nn as nn
import modules.rrdb_denselayer
from modules.hinet import Hinet


class PRIS(nn.Module):
    def __init__(self, in_1=3, in_2=3):
        super(PRIS, self).__init__()
        self.inbs = Hinet(in_1=in_1, in_2=in_2)
        self.pre_enhance = modules.rrdb_denselayer.ResidualDenseBlock_out(3, 3)
        self.post_enhance = modules.rrdb_denselayer.ResidualDenseBlock_out(3, 3)

    def load_hinet(self, path):
        state_dicts = torch.load(path)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.hinet.load_state_dict(network_state_dict)

    def forward(self, x, rev=False):

        if not rev:
            out = self.hinet(x)

        else:
            out = self.hinet(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)