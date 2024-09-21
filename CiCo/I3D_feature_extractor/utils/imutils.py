import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont

from .misc import to_numpy, to_torch


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def im_to_video(img):
    assert img.dim() == 3
    nframes = int(img.size(0) / 3)
    return img.contiguous().view(3, nframes, img.size(1), img.size(2))


def video_to_im(video):
    assert video.dim() == 4
    assert video.size(0) == 3
    return video.view(3 * video.size(1), video.size(2), video.size(3))


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode="RGB"))


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print(("%f %f" % (img.min(), img.max())))
    img = scipy.misc.imresize(img, (oheight, owidth))
    img = im_to_torch(img)
    print(("%f %f" % (img.min(), img.max())))
    return img


def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img


def imshow(img):
    npimg = im_to_numpy(img * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis("off")


def rectangle_on_image(img, width=5, frame_color="yellow"):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    cor = (0, 0, img_pil.size[0], img_pil.size[1])
    for i in range(width):
        draw.rectangle(cor, outline=frame_color)
        cor = (cor[0] + 1, cor[1] + 1, cor[2] - 1, cor[3] - 1)
    return np.asarray(img_pil)


def text_on_image(img, txt=""):
    x = 5
    y = 5
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_name = "FreeSerif.ttf"
    font_name = "DejaVuSerif.ttf"
    font_name = "DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_name, int(img.shape[0] / 8))
    w, h = font.getsize(txt)
    if w - 2 * x > img.shape[0]:
        font = ImageFont.truetype(
            font_name, int(img.shape[0] * img.shape[0] / (8 * (w - 2 * x)))
        )
        w, h = font.getsize(txt)
    draw.rectangle((x, y, x + w, y + h), fill="black")
    draw.text((x, y), txt, fill=(255, 255, 255), font=font)
    return np.asarray(img_pil)
