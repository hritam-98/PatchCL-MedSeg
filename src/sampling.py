from abc import abstractmethod

import torch


class Sampler:
    def __init__(self, n: int = 10) -> None:
        self.n = n

    def preprocess(self, values: torch.Tensor) -> torch.Tensor:
        return values  # Defaults to noop

    @abstractmethod
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        ...


class UniformSampler(Sampler):
    def sample(self, values: torch.Tensor) -> torch.Tensor:
        return torch.randint(low=0, high=values.shape[-1], size=(values.shape[0], 2, values.shape[1], self.n))


class ProbabilisticSampler(Sampler):
    def __init__(self, n: int = 10, alpha: float = 1) -> None:
        super().__init__(n)
        self.alpha = alpha

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        maxes = values.max(dim=-1, keepdim=True).values
        normalised_values = values / maxes

        exponentiated_values = normalised_values ** self.alpha

        sums = exponentiated_values.sum(dim=-1, keepdim=True)
        probabilities_positives = exponentiated_values / sums
        probabilities_negatives = (1 - probabilities_positives) / (probabilities_positives.shape[-1] - 1)

        indices_positives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_positives
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_negatives
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class ProbabilisticSentinelSampler(Sampler):
    def __init__(self, n: int = 10, alpha: float = 1) -> None:
        super().__init__(n)
        self.alpha = alpha

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        sentinels_added = torch.dstack((values, torch.ones(values.shape[:-1], device=values.device)))

        exponentiated_values = sentinels_added ** self.alpha

        sums = exponentiated_values.sum(dim=-1, keepdim=True)
        probabilities_positives = exponentiated_values / sums
        probabilities_negatives = (1 - probabilities_positives) / (probabilities_positives.shape[-1] - 1)

        indices_positives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_positives
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.multinomial(image, self.n, replacement=True)
                for image in probabilities_negatives
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class TopKSampler(Sampler):
    def __init__(self, n: int = 10, k: int = 50) -> None:
        super().__init__(n)
        self.k = k

    def sample(self, values: torch.Tensor) -> torch.Tensor:
        top_k_positive_indices = torch.topk(values, k=self.k, dim=-1).indices
        top_k_negative_indices = torch.topk(-values, k=values.shape[-1] - self.k, dim=-1).indices

        indices_positives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n,))]
                        for channel in image
                    ]
                )
                for image in top_k_positive_indices
            ]
        )
        indices_negatives = torch.stack(
            [
                torch.stack(
                    [
                        channel[torch.randint(low=0, high=channel.numel(), size=(self.n,))]
                        for channel in image
                    ]
                )
                for image in top_k_negative_indices
            ]
        )

        return torch.stack((indices_positives, indices_negatives)).permute(1, 0, 2, 3)


class EntropySampler(Sampler):
    def preprocess(self, values: torch.Tensor) -> torch.Tensor:
        def clamp(x):
            return torch.maximum(x, torch.tensor(-100))
        ce = -(values * clamp(torch.log2(values)) + (1 - values) * clamp(torch.log2(1 - values)))
        return ce

    def sample(self, entropies: torch.Tensor, attentions: torch.Tensor) -> torch.Tensor:
        sampled = torch.stack(
            [
                torch.multinomial(torch.maximum(1 - image, torch.tensor(0.001)), self.n, replacement=True)
                for image in entropies
            ]
        )
        parity = torch.bernoulli(attentions)
        positives = [
            [
                [i for i in channel if parity[image_idx, channel_idx, i] == 1]
                for channel_idx, channel in enumerate(image)
            ]
            for image_idx, image in enumerate(sampled)
        ]
        negatives = [
            [
                [i for i in channel if parity[image_idx, channel_idx, i] == 0]
                for channel_idx, channel in enumerate(image)
            ]
            for image_idx, image in enumerate(sampled)
        ]
        return list(zip(positives, negatives))
