from argparse import ArgumentParser

import src.data
import src.util.arguments
import src.util.models
import src.util.plot
import torch
from scipy import ndimage
from skimage import morphology
from torch.utils.data import DataLoader, Dataset


def morphological_operations(mask: torch.Tensor) -> torch.Tensor:
    """Improve segmentation masks slightly by a couple of opening and closing operations"""
    mask = mask.numpy()
    mask = ndimage.binary_opening(mask, structure=morphology.disk(3)).astype(int)
    mask = ndimage.binary_closing(mask, structure=morphology.disk(3)).astype(int)
    mask = ndimage.binary_opening(mask, structure=morphology.disk(1)).astype(int)
    mask = ndimage.binary_closing(mask, structure=morphology.disk(1)).astype(int)
    return torch.tensor(mask)


def choose_model_checkpoint(version_number: int | str):
    log = src.util.models.load_log(version_number)
    best_epoch_info = max(log, key=lambda e: float(e['best_val_mdc']) if e['best_val_mdc'] else -1)

    epoch = int(best_epoch_info['epoch'])
    step = int(best_epoch_info['step'])

    # Find the channel which performed the best, again according to validation dice score
    n_classes = len([k for k in best_epoch_info.keys() if k.startswith('mdc_c')])
    channel = max(range(n_classes), key=lambda i: best_epoch_info[f'mdc_c{i+1}'])

    validation_score = float(best_epoch_info['best_val_mdc'])

    return epoch, step, channel, validation_score


def evaluate_performance(model: torch.nn.Module, channel: int, dataset: Dataset, batch_size: int = 10):
    scores = []

    data_loader = DataLoader(dataset, batch_size=batch_size)
    for images, masks in data_loader:
        with torch.no_grad():
            confidence_maps = model(images)
        predicted_masks = confidence_maps > 0.5

        # Extract the prediction from the selected class/channel
        channel_masks = masks[:, 0]
        predicted_masks = predicted_masks[:, channel]

        # Perform basic morphological operations
        predicted_masks = torch.stack(list(map(morphological_operations, predicted_masks)))

        # Compute dice score
        tp = torch.logical_and(channel_masks > 0.5, predicted_masks > 0.5).flatten(start_dim=1).sum(dim=1)
        fp_fn = torch.logical_xor(channel_masks > 0.5, predicted_masks > 0.5).flatten(start_dim=1).sum(dim=1)

        dice_scores = 2 * tp / (2 * tp + fp_fn)

        scores.extend([image_score.item() for image_score in dice_scores])

    return sum(scores) / len(scores)


def main():
    parser = ArgumentParser()
    dataset = src.util.arguments.add_options(parser, 'dataset', list(src.data.available_validation_datasets))
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--versions', type=int, nargs='+')
    args = parser.parse_args()

    print('Evaluating each model...')
    for version_number in args.versions:
        print(f'Version {version_number}')
        epoch, step, channel, _ = choose_model_checkpoint(version_number)
        model = src.util.models.load_model(version_number, epoch, step)

        score = evaluate_performance(model, channel, dataset, args.batch_size)

        print('DICE score:', score, '\tfrom version', version_number, 'epoch', epoch, 'channel', channel)

    print('Choosing model (from validation scores)...')
    version, (epoch, step, channel, validation_score) = max(((version, choose_model_checkpoint(version)) for version in args.versions), key=lambda e: e[1][3])
    print(f'Chose version {version} epoch {epoch}. Had validation score {validation_score}')
    model = src.util.models.load_model(version, epoch, step)

    images, masks = map(torch.stack, zip(*list(dataset)))

    # Predict on test set
    with torch.no_grad():
        confidence_maps = model(images)[:, channel]
        predictions = torch.stack(list(map(morphological_operations, (confidence_maps > 0.5).type(torch.int))))

    # Plot predictions on test set
    src.util.plot.plot_mask_2(
        images=images,
        masks=masks,
        attention_maps=confidence_maps.unsqueeze(1),
        predictions=predictions.unsqueeze(1),
        path='test_best_model_predictions.png',
        dpi=900
    )


if __name__ == '__main__':
    main()
