from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import src.util.events
from src.sampling import Sampler
from src.similarity import SimilarityMeasure

# Indices
POSITIVE = 0
NEGATIVE = 1


def sample(population: Union[Sequence, torch.Tensor], k: int, replace=False):
    n = len(population)
    indices = np.random.choice(n, k, replace=replace)
    return [population[i] for i in indices]


class Counter(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.f = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, bias=False)
        w = torch.stack(
            [
                torch.stack(
                    [
                        (torch.ones if i == j else torch.zeros)(kernel_size, kernel_size, dtype=torch.float)
                        for j in range(channels)
                    ]
                )
                for i in range(channels)
            ]
        )
        self.f.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x) / self.kernel_size ** 2

    def count(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        counts: torch.Tensor = self(x)
        return counts.flatten(start_dim=-2), counts.shape


@dataclass
class LoggedInfo:
    epoch: int
    batch_idx: int
    log_dir: str
    images: torch.Tensor
    attention_maps: torch.Tensor
    attended_images: torch.Tensor = None
    regions: torch.Tensor = None
    sampled_regions: torch.Tensor = None
    unpack_index: Callable = None
    kernel_size: int = None
    masks: torch.Tensor = None


class Model(pl.LightningModule):
    def __init__(
        self,
        counter: Counter,
        sampler: Sampler,
        similarity_measure: SimilarityMeasure,
        confidence_network: nn.Module,
        featuriser_network: nn.Module,
        inter_channel_loss_scaling_factor: float = 1,
        learning_rate: float = 0.0002,
        info_to_log: dict[str, str] = {},  # logged as hyperparameters by self.save_hyperparameters call
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=['counter', 'attention_network', 'feature_network'])

        self.counter = counter
        self.sampler = sampler
        self.similarity_measure = similarity_measure

        self.confidence_network = confidence_network
        self.featuriser_network = featuriser_network

        self.inter_channel_loss_scaling_factor = inter_channel_loss_scaling_factor

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.confidence_network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images = batch
        attention_maps = self.confidence_network(images)
        attended_images = torch.stack([images[:, i, None] * attention_maps for i in range(images.shape[1])], dim=2)

        n_images = images.shape[0]
        n_classes = attention_maps.shape[1]

        regions, count_shape = self.counter.count(self.sampler.preprocess(attention_maps))
        try:
            sampled_regions = self.sampler.sample(regions, self.counter.count(attention_maps)[0])
        except TypeError:
            sampled_regions = self.sampler.sample(regions)
        # sampled_regions is of shape: image, parity, channel, region

        def unpack_index(idx):
            idx = idx.item()
            row = idx // count_shape[-1]
            col = idx % count_shape[-1]
            return row * self.counter.stride, col * self.counter.stride

        # extract regions
        x = [([], []) for _ in range(n_classes)]
        for attended_image, (positive_regions, negative_regions) in zip(attended_images, sampled_regions):
            for parity, parity_regions in [(POSITIVE, positive_regions), (NEGATIVE, negative_regions)]:
                for channel_index, channel_regions in enumerate(parity_regions):
                    x[channel_index][parity].extend((
                        attended_image[channel_index, :, row:row + self.counter.kernel_size, col:col + self.counter.kernel_size]
                        for row, col in map(unpack_index, channel_regions)
                        if row < attended_image.shape[-2] - self.counter.kernel_size  # disregard sentinels
                    ))

        if any(any(len(p) == 0 for p in c) for c in x):
            return None

        # vectorise within class, parity
        y = [tuple(map(torch.stack, c)) for c in x]

        if self.similarity_measure.featurised:
            # featurise regions; shape class, parity, prediction
            y = [tuple(map(self.featuriser_network, c)) for c in y]

        # contrast regions:
        loss = torch.zeros(1, device=batch.device)
        for c in range(n_classes):
            # select positive representatives from this class
            positives = sample(y[c][POSITIVE], n := 10, replace=True)

            # Intra-channel contrastive loss:
            intra_channel_pos_pos_similarity = [
                self.similarity_measure.similarity(positive, y[c][POSITIVE])
                for positive in positives
            ]
            intra_channel_pos_neg_similarity = [
                self.similarity_measure.similarity(positive, y[c][NEGATIVE])
                for positive in positives
            ]
            intra_channel_contrastive_loss = sum(
                -torch.log(torch.exp(pp) / torch.exp(pn).sum()).sum()
                for pp, pn in zip(intra_channel_pos_pos_similarity, intra_channel_pos_neg_similarity)
            ) / n**2 / n_images
            self.log("intra", intra_channel_contrastive_loss)
            loss += intra_channel_contrastive_loss

            # Inter-channel contrastive loss:
            if n_classes > 1:  # Only makes sense if more than one channel
                # All the positive samples from all the other classes
                all_other_classes_positives = torch.vstack([y[i][POSITIVE] for i in range(len(y)) if i != c])

                inter_channel_pos_pos_similarity = [
                    self.similarity_measure.similarity(positive, all_other_classes_positives)
                    for positive in positives
                ]
                inter_channel_contrastive_loss = sum(
                    -torch.log(torch.exp(intra_pp) / torch.exp(inter_pp).sum()).sum()
                    for intra_pp, inter_pp in zip(intra_channel_pos_pos_similarity, inter_channel_pos_pos_similarity)
                ) / n**2 / n_images
                self.log("inter", inter_channel_contrastive_loss)
                loss += self.inter_channel_loss_scaling_factor * inter_channel_contrastive_loss

        # Facilitate external plotting
        src.util.events.post_event(
            src.util.events.EventTypes.END_OF_TRAINING_BATCH,
            data=LoggedInfo(
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                log_dir=self.logger.log_dir,
                images=images.detach(),
                attention_maps=attention_maps.detach(),
                attended_images=attended_images.detach(),
                regions=regions.detach(),
                sampled_regions=sampled_regions,
                unpack_index=unpack_index,
                kernel_size=self.counter.kernel_size,
            )
        )

        if loss == 0:
            return None
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, masks = batch
            attention_maps = self.confidence_network(images)
            predicted_masks = attention_maps > 0.5

        masks = masks > 0.5  # turn it into a boolean tensor
        tp = torch.logical_and(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)
        fp_fn = torch.logical_xor(masks, predicted_masks).flatten(start_dim=2).sum(dim=2)

        dice_scores = 2 * tp / (2 * tp + fp_fn)

        for image_scores in dice_scores:
            for i, channel_score in enumerate(image_scores, start=1):
                self.log(f"mdc_c{i}", channel_score)

        src.util.events.post_event(
            src.util.events.EventTypes.END_OF_VALIDATION_BATCH,
            data=LoggedInfo(
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                log_dir=self.logger.log_dir,
                images=images,
                attention_maps=attention_maps,
                masks=masks
            )
        )

        return dice_scores

    def validation_epoch_end(self, validation_step_outputs):
        scores = torch.vstack(validation_step_outputs)
        self.log("best_val_mdc", max(scores.mean(0)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        return optimizer
