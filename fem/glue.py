import torch
from torch import nn
import numpy

message_size = 64
embed_shape = 256

class PosEmbedderMLP(nn.Module):
    def __init__(self, point_size=2):
        super().__init__()
        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.Linear(20, 64)
        self.l3 = nn.Linear(64, 256)

    def forward(self, points):
        todo: add batchnorm
        x = nn.functional.relu(self.l1(points))
        x = nn.functional.relu(self.l2(x))
        x = nn.functional.relu(self.l3(x))
        return x

class MessageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(embed_shape + message_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = nn.functional.relu(self.l3(x))
        return x


class Glue(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = PosEmbedderMLP()
        self.key_size = 32
        self.value_size = 64
        self.projection = nn.Linear(256, self.key_size + self.value_size)
        # query must match key size
        self.query_projection = nn.Linear(256, self.key_size)
        self.message_mlp = MessageMLP()
        self.final_projection = nn.Linear(256, 256)
        self.bin_score = torch.Tensor(1)

    def key_value(self, layer_embedding):
        key_value = self.projection(layer_embedding)
        k = key_value[:, :self.key_size]
        v = key_value[:, self.key_size:]
        return k, v

    def augment_score(self, S):
        height, width = 1, 2
        S = torch.cat([S, (torch.ones(S.shape[height]) * self.bin_score).unsqueeze(0)],
                          dim=height + 1)
        S = torch.cat([S, (torch.ones(S.shape[width] + 1) * self.bin_score).unsqueeze(0)],
                          dim=width + 1)
        return S

    def forward(self, points1, points2, desc1, desc2):
        import pdb;pdb.set_trace()
        points = torch.cat([points1, points2], dim=0).float()
        desc = torch.cat([desc1, desc2])
        x = desc + self.pos_embed(points)
        for i in range(0, 5):
            x = self.compute_new_state(len(points1), x, i % 2)
        f = self.final_projection(x)
        score = f[:len(points1)].dot(f[len(points2):])
        S = self.augment_score(score)
        import pdb;pdb.set_trace()
        r = torch.ones(10)

    def compute_new_state(self, n_source, x0, cross):
        query = self.query_projection(x0)
        k0, v0 = self.key_value(x0)
        message_s, message_t = self.message(k0, n_source, query, v0, cross=cross)
        # stack messages
        message = torch.cat([message_s, message_t])
        x1 = x0 + self.message_mlp(torch.cat([x0, message], dim=1))
        return x1

    def message(self, k0, n_source, query, v0, cross=True):
        query = query / query.min()
        k0 = k0 / k0.min()
        v0 = v0 / v0.min()
        # split by images
        k0_s, k0_t = k0[:n_source], k0[n_source:]
        v0_s, v0_t = v0[:n_source], v0[n_source:]
        q0_s, q0_t = query[:n_source], query[n_source:]
        # use self attention for layer 0
        # todo: not sure if q_i @ k_i should be removed or not
        import pdb;pdb.set_trace()

        if cross:
            alfa_s = nn.functional.softmax(q0_s @ k0_t.T, dim=1)
            alfa_t = nn.functional.softmax(q0_t @ k0_s.T, dim=1)
            message_s = alfa_s @ v0_t
            message_t = alfa_t @ v0_s
        else:
            alfa_s = nn.functional.softmax(q0_s @ k0_s.T, dim=1)
            alfa_t = nn.functional.softmax(q0_t @ k0_t.T, dim=1)
            message_s = alfa_s @ v0_s
            message_t = alfa_t @ v0_t
        return message_s, message_t


def test():
    g = Glue()
    points1 = [
                                              [[10, 20],
                                              [18, 43]],
                                              [[12, 14],
                                              [44, 45]]
    ]
    points2 = [[[10, 20],
              [18, 43],
              [129, 232]],
             [[84, 23],
              [44, 23]]]

    desc1 = torch.Tensor(2 * 256).reshape((2, 256)) + 20
    desc2 = torch.Tensor(3 * 256).reshape((3, 256)) + 20
    g.forward(torch.from_numpy(numpy.asarray(points1[0])),
              torch.from_numpy(numpy.asarray(points2[0])), desc1, desc2)


test()
