import torch
from fem.sinkhorn import optimal_transport
from fem.util import init_weights
from torch import nn
import numpy

message_size = 64
embed_shape = 256
dust = -1


class PosEmbedderMLP(nn.Module):
    def __init__(self, point_size=2):
        super().__init__()
        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.Linear(20, 64)
        self.l3 = nn.Linear(64, 256)
        init_weights(self)

    def forward(self, points):
        x = (nn.functional.relu(self.l1(points)))
        x = (nn.functional.relu(self.l2(x)))
        x = (nn.functional.relu(self.l3(x)))
        return x


class MessageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(embed_shape + message_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        init_weights(self)
    def forward(self, x):
        x = (nn.functional.relu(self.l1(x)))
        x = (nn.functional.relu(self.l2(x)))
        x = (nn.functional.relu(self.l3(x)))
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
        init_weights(self)

    def key_value(self, layer_embedding):
        key_value = self.projection(layer_embedding)
        k = key_value[:, :self.key_size]
        v = key_value[:, self.key_size:]
        return k, v

    def augment_score(self, S):
        height, width = 0, 1
        S = torch.cat([S, (torch.ones(S.shape[width]) * self.bin_score).unsqueeze(0)], dim=height)
        S = torch.cat([S, (torch.ones(S.shape[height]) * self.bin_score).unsqueeze(0).T], dim=width)
        return S

    def forward(self, data):
        points1 = data.get('points1')
        points2 = data.get('points2')
        desc1 = data.get('desc1')
        desc2 = data.get('desc2')
        conf1 = data.get('conf1')
        conf2 = data.get('conf2')
        points = torch.cat([points1, points2], dim=0).float()
        desc = torch.cat([desc1, desc2])
        x = desc + self.pos_embed(points)
        for i in range(0, 5):
            x = self.compute_new_state(len(points1), x, i % 2)
        f = self.final_projection(x)
        score = f[:len(points1)] @ (f[len(points1):]).T
        S = self.augment_score(score)
        len_source = len(points1) + 1
        len_target = len(points2) + 1
        # source, target + dustbin
        r_source = torch.ones(len_source)
        c_target = torch.ones(len_target)
        source_dust = len_target - 1
        r_source[dust] = source_dust
        target_dust = len_source - 1
        c_target[dust] = target_dust
        P, cost = optimal_transport(S, r_source, c_target, 5, epsilon=0.01)
        return {'P': P, 'cost': cost}

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
    data = {'points1': torch.from_numpy(numpy.asarray(points1[0])),
            'points2': torch.from_numpy(numpy.asarray(points2[0])),
            'desc1': desc1,
            'desc2': desc2}
    g.forward(data)


test()
