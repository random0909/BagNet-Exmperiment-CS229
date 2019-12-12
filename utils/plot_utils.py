import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform
import torch

# reference: https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/utils.py
def plot_heatmap(heatmap, original, ax, cmap='RdBu_r', percentile=99, dilation=0.5, alpha=0.25):
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        original_greyscale = original if len(original.shape) == 2 else np.mean(original, axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant',
                                              multichannel=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)


def generate_heatmap_pytorch(model, image, target, patchsize):
    with torch.no_grad():
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize - 1) // 2:(patchsize - 1) // 2 + x, (patchsize - 1) // 2:(patchsize - 1) // 2 + y] = image[0]
        image = padded_image[None].astype(np.float32)

        patches = torch.from_numpy(image).cuda().permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        logits_list = []
        for batch_patches in torch.split(patches, 1000):
            logits = model(batch_patches)
            logits = logits[:, target][:, 0]
            logits_list.append(logits.data.cpu().numpy().copy())
        logits = np.hstack(logits_list)
        return logits.reshape((224, 224))
