import torch


def init_weights(self):
    def init_weights(m):
        if hasattr(m, 'weight') and not isinstance(m, (torch.nn.BatchNorm2d,
                                                       torch.nn.BatchNorm1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    self.apply(init_weights)
