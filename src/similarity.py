from abc import abstractmethod
import torch
import torch.nn as nn
from kornia.enhance.histogram import histogram2d
from torchvision.transforms.transforms import Grayscale


class SimilarityMeasure:
    featurised: bool = False

    @abstractmethod
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        ...


class CrossEntropy(SimilarityMeasure):
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
        assert len(x1.shape) == 3
        ce = -(x1 * torch.log2(x2) + (1 - x1) * torch.log2(1 - x2))
        return -ce.mean(dim=(-1, -2, -3))


class MeanSquaredError(SimilarityMeasure):
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
        assert len(x1.shape) == 3
        mse = (x1 - x2) ** 2
        return -mse.mean(dim=(-1, -2, -3))


class FeaturisedCosine(SimilarityMeasure):
    featurised = True
    cosine_similarity = nn.CosineSimilarity(dim=1)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
        assert len(x1.shape) == 1
        return self.cosine_similarity(x1, x2)


class MutualInformation(SimilarityMeasure):
    grey_scale = Grayscale(1)

    def _mi(self, v1, v2):
        v1 = self.grey_scale(v1).flatten(start_dim=1)
        v2 = self.grey_scale(v2).flatten(start_dim=1)
        joint_histogram = histogram2d(v1, v2, bins=torch.linspace(0, 1, 25, device=v1.device), bandwidth=torch.tensor(0.9))
        p_xy = joint_histogram / joint_histogram.sum()
        p_x = p_xy.sum(dim=2)
        p_y = p_xy.sum(dim=1)
        p_x_p_y = p_x[:, :, None] * p_y[:, None, :]
        mi = (p_xy * torch.log2(p_xy / p_x_p_y)).sum(dim=(1, 2))
        return mi

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
        assert len(x1.shape) == 3
        if len(x2.shape) == 3:
            x2 = x2.unsqueeze(0)

        b = x2.shape[0]
        x1 = x1.repeat(b, 1, 1).reshape(b, *x1.shape)
        return self._mi(x1, x2)


class KLDivergence(SimilarityMeasure):
    grey_scale = Grayscale(1)
    kl_divergence = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape or x1.shape == x2.shape[1:]
        assert len(x1.shape) == 3
        if len(x2.shape) == 3:
            x2 = x2.unsqueeze(0)
        v1 = torch.maximum(self.grey_scale(x1).flatten().log(), torch.tensor(-100))
        v2 = torch.maximum(self.grey_scale(x2).flatten(start_dim=1).log(), torch.tensor(-100))
        divergence = torch.stack([self.kl_divergence(v1, v) for v in v2])
        return -divergence
