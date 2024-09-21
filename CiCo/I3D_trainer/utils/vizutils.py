import getpass
from os.path import dirname

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
from PIL import Image

from utils.imutils import rectangle_on_image, text_on_image

from .misc import mkdir_p, to_numpy


def _imshow_pytorch(rgb, ax=None):
    from utils.transforms import im_to_numpy

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(im_to_numpy(rgb * 255).astype(np.uint8))
    ax.axis("off")


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf


def viz_sample(img, out_torch, class_names=None, n=None, target=None):
    # Note: use class_names from the first batch because they are duplicates.
    # It might become an issue when several datasets are concatenated,
    # But usually same vocab are concatenated
    if isinstance(out_torch, torch.FloatTensor):  # Prediction
        out = torch.nn.functional.softmax(out_torch, dim=0).data
        v, out = torch.max(out, 0)
        out = out.item()
        frame_color = "green" if target[n] == out else "red"
        img = text_on_image(img, txt=class_names[out][0])
        img = rectangle_on_image(img, frame_color=frame_color)
    else:  # Ground truth
        out = out_torch
        img = text_on_image(img, txt=class_names[out][0])
    return img


def viz_batch(
    inputs,
    outputs,
    mean=torch.Tensor([0.5, 0.5, 0.5]),
    std=torch.Tensor([1.0, 1.0, 1.0]),
    num_rows=2,
    parts_to_show=None,
    supervision=0,
    meta=None,
    target=None,
    save_path="",
    pose_rep="vector",
):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n]
        # Un-normalize
        inp = (inp * std.view(3, 1, 1).expand_as(inp)) + mean.view(3, 1, 1).expand_as(
            inp
        )
        # Torch to numpy
        inp = to_numpy(inp.clamp(0, 1) * 255)
        inp = inp.transpose(1, 2, 0).astype(np.uint8)
        # Resize 256x256 to 512x512 to be bigger
        # inp = scipy.misc.imresize(inp, [256, 256])
        batch_img.append(
            viz_sample(
                inp, outputs[n], class_names=meta["class_names"], n=n, target=target
            )
        )
    return np.concatenate(batch_img)


def viz_gt_pred(
    inputs,
    outputs,
    target,
    mean,
    std,
    meta,
    gt_win,
    pred_win,
    fig,
    save_path=None,
    show=False,
):
    if save_path is not None:
        mkdir_p(dirname(save_path))
    # Save video viz
    if inputs[0].dim() == 4:
        # In case GPU preprocessing was used, we copy to the CPU
        data = inputs.cpu()
        suffix = ".avi"
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        nframes = inputs.size(2)
        out = cv2.VideoWriter(str(save_path) + suffix, fourcc, 10, (1000, 1000))

        # For each frame
        for t in range(0, nframes, 1):
            inp = data[:, :, t, :, :]
            # This was used to save the original-res inputs for the supmat
            if False:
                # e.g. inp size [batch, 3, 224, 224]
                # for each batch
                for b in range(inp.shape[0]):
                    class_name = meta["class_names"][target[b]][0].replace(" ", "_")
                    img_file = f"{save_path}_{class_name}_{t}_{b}.png"
                    np_img = to_numpy((inp[b] + 0.5).clamp(0, 1) * 255)
                    np_img = np_img.transpose(1, 2, 0).astype(np.uint8)
                    np_img = np_img[:, :, ::-1]
                    cv2.imwrite(img_file, np_img)
            gt_win, pred_win, fig = viz_gt_pred_single(
                inp,
                outputs,
                target,
                mean,
                std,
                meta,
                gt_win,
                pred_win,
                fig,
                save_path,
                show,
            )
            fig_img = fig2data(fig)
            # fig_img = scipy.misc.imresize(fig_img, [1000, 1000])
            fig_img = np.array(Image.fromarray(fig_img).resize([1000, 1000]))
            out.write(fig_img[:, :, (2, 1, 0)])
        out.release()
    return gt_win, pred_win, fig


def viz_gt_pred_single(
    inputs, outputs, target, mean, std, meta, gt_win, pred_win, fig, save_path, show
):
    # print(inputs[:, 0, :5, 0])  # inputs: [10, 3, 256, 256]
    gt_batch_img = viz_batch(
        inputs, target, mean=mean, std=std, meta=meta, save_path=dirname(save_path)
    )
    pred_batch_img = viz_batch(
        inputs,
        outputs,
        mean=mean,
        std=std,
        meta=meta,
        target=target,
        save_path=dirname(save_path),
    )
    if not gt_win or not pred_win:
        fig = plt.figure(figsize=(20, 20))
        ax1 = plt.subplot(121)
        ax1.title.set_text("Groundtruth")
        gt_win = plt.imshow(gt_batch_img)
        ax2 = plt.subplot(122)
        ax2.title.set_text("Prediction")
        pred_win = plt.imshow(pred_batch_img)
    else:
        gt_win.set_data(gt_batch_img)
        pred_win.set_data(pred_batch_img)

    if show:
        print("Showing")
        plt.pause(0.05)

    return gt_win, pred_win, fig
