from argparse import ArgumentParser

import numpy as np
import src.data
import src.util.arguments
import src.util.plot
import torch


def dice_score(predictions, masks):
    tp = torch.logical_and(masks, predictions).flatten(start_dim=2).sum(dim=2)
    fp_fn = torch.logical_xor(masks, predictions).flatten(start_dim=2).sum(dim=2)

    dice_scores = 2 * tp / (2 * tp + fp_fn)
    return dice_scores.mean().item()


def main():
    parser = ArgumentParser()
    train_dataset = src.util.arguments.add_options(parser, 'train_dataset', list(src.data.available_validation_datasets))
    test_dataset = src.util.arguments.add_options(parser, 'test_dataset', list(src.data.available_validation_datasets))

    images, masks = map(torch.stack, zip(*list(train_dataset)))

    optimal_threshold = max(np.linspace(0, 1, 100), key=lambda t: dice_score(images < t, masks > 0.5))
    print(optimal_threshold)
    print(dice_score(images < optimal_threshold, masks > 0.5))

    images, masks = map(torch.stack, zip(*list(test_dataset)))
    print(dice_score(images < optimal_threshold, masks > 0.5))
    src.util.plot.plot_mask(images, masks, (images < optimal_threshold).type(torch.float), path='optimal_threshold.png')


if __name__ == '__main__':
    main()
