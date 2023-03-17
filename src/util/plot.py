import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def plotable(image: torch.Tensor):
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[0] == 1:
            return image.squeeze(0)
        elif image.shape[0] == 3:
            return image.permute((1, 2, 0))
        else:
            raise RuntimeError
    else:
        raise RuntimeError


def plot_selected_crops(data, path=None, dpi=300):
    if len(data) <= 1:
        return

    n = min(len(data), 10)
    n_channels = len(data[1][1])
    fig, axs = plt.subplots(
        1 + 2 * n_channels, n, figsize=(n, 1 + 2 * n_channels)
    )
    for i, (image, attention_map, attended_image, positive_regions, negative_regions, size) in enumerate(data[:n]):
        axs[0][i].imshow(plotable(image))
        for j in range(n_channels):
            axs[1 + j][i].imshow(plotable(attention_map[j]))
            axs[1 + n_channels + j][i].imshow(plotable(attended_image[j]))

        for j in range(1, 1 + 2 * n_channels):
            for row, col in positive_regions[(j - 1) % n_channels]:
                axs[j][i].add_patch(
                    Rectangle(
                        (col, row),
                        size,
                        size,
                        linewidth=0.5,
                        edgecolor='white',
                        facecolor="none",
                    )
                )
            for row, col in negative_regions[(j - 1) % n_channels]:
                axs[j][i].add_patch(
                    Rectangle(
                        (col, row),
                        size,
                        size,
                        linewidth=0.5,
                        edgecolor='red',
                        facecolor="none",
                    )
                )

        for j in range(1 + 2 * n_channels):
            axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    plt.close()


markings = np.array([(0, 0, 0, 0), (0, 255, 50, 255)])
contours = np.array([(0, 0, 0, 0), (0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255)])


def plot_mask(images, masks, attention_maps, path=None, dpi=300):
    assert len(images) == len(masks) == len(attention_maps)
    n = len(images)
    n_channels = len(attention_maps[1])
    fig, axs = plt.subplots(2 + 2 * n_channels, n, figsize=(n, 2 + 2 * n_channels))
    for i, (image, mask, attention_map) in enumerate(zip(images, masks, attention_maps)):
        axs[0][i].imshow(plotable(image))

        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask = mask.type(torch.IntTensor)
        axs[1][i].imshow(markings[mask])

        for j, attention_map_channel in enumerate(attention_map, start=0):
            axs[2 * j + 2][i].imshow(plotable(attention_map_channel))
            diffs = (attention_map_channel > 0.5).squeeze(0).type(torch.IntTensor)
            diffs[torch.logical_and(diffs, 1 - mask)] = 3
            diffs[torch.logical_and(1 - diffs, mask)] = 2
            axs[2 * j + 3][i].imshow(contours[diffs])

        for j in range(2 + 2 * n_channels):
            axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    plt.close()


def plot_mask_2(images, masks, attention_maps, predictions, path=None, dpi=300):
    assert len(images) == len(masks) == len(attention_maps)
    n = len(images)
    n_channels = len(attention_maps[1])
    fig, axs = plt.subplots(2 + 2 * n_channels, n, figsize=(n, 2 + 2 * n_channels))
    for i, (image, mask, attention_map, prediction) in enumerate(zip(images, masks, attention_maps, predictions)):
        axs[0][i].imshow(plotable(image))

        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask = mask.type(torch.IntTensor)
        axs[1][i].imshow(markings[mask])

        for j, (attention_map_channel, prediction_channel) in enumerate(zip(attention_map, prediction), start=0):
            axs[2 * j + 2][i].imshow(plotable(attention_map_channel))
            # diffs = (attention_map_channel > 0.5).squeeze(0).type(torch.IntTensor)
            diffs = prediction_channel
            diffs[torch.logical_and(diffs, 1 - mask)] = 3
            diffs[torch.logical_and(1 - diffs, mask)] = 2
            axs[2 * j + 3][i].imshow(contours[diffs])

        for j in range(2 + 2 * n_channels):
            axs[j][i].axis("off")

    if path is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    plt.close()
