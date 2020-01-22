from collections import defaultdict


class Stats:
    def __init__(self, **coeff):
        self.stats = defaultdict(float)
        self.coeff = defaultdict(lambda: 0.05)
        self.coeff.update(coeff)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.stats:
                self.stats[k] = v
            self.stats[k] = self.stats[k] * (1 - self.coeff[k]) + v * self.coeff[k]

    def __str__(self):
        return str(self.stats)
