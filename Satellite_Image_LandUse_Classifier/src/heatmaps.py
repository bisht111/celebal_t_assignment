import numpy as np
import matplotlib.pyplot as plt


def pixel_difference(before, after):
    """
    Pixel-wise absolute RGB difference.
    """

    before = np.asarray(before).astype(np.float32)
    after = np.asarray(after).astype(np.float32)

    diff = np.abs(before - after)

    heatmap = diff.mean(axis=2)

    return heatmap


def visualize_change(before,
                     after,
                     similarity,
                     threshold):

    heatmap = pixel_difference(before, after)

    changed = similarity < threshold

    fig, ax = plt.subplots(1,3, figsize=(14,5))

    ax[0].imshow(before)
    ax[0].set_title("Before (T1)")
    ax[0].axis("off")

    ax[1].imshow(after)
    ax[1].set_title("After (T2)")
    ax[1].axis("off")

    im = ax[2].imshow(
        heatmap,
        cmap="hot"
    )

    ax[2].set_title("Change Heatmap")
    ax[2].axis("off")

    plt.colorbar(im)

    plt.suptitle(
        f"Cosine Similarity = {similarity:.3f}\n"
        f"Threshold = {threshold:.3f}\n"
        f"Changed = {changed}"
    )

    plt.tight_layout()
    plt.show()