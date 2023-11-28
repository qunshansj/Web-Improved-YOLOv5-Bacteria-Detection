python
import torch

class WassersteinLoss:
    def __init__(self, eps=1e-7, constant=12.8):
        self.eps = eps
        self.constant = constant

    def __call__(self, pred, target):
        center1 = pred[:, :2]
        center2 = target[:, :2]

        whs = center1[:, :2] - center2[:, :2]

        center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + self.eps

        w1 = pred[:, 2]  + self.eps
        h1 = pred[:, 3]  + self.eps
        w2 = target[:, 2] + self.eps
        h2 = target[:, 3] + self.eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wasserstein_2 = center_distance + wh_distance
        return torch.exp(-torch.sqrt(wasserstein_2) / self.constant)
