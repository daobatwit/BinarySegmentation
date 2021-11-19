import os

import albumentations as album
import cv2

from util import keep_largest_comp, region_fill_mask, keep_both_joints

# run iteration
# this variable is used to conveniently change the namings of
# output files/directories
ITR = 9

# dataset directories
train_dir = 'data\\training\\'
test_dir = 'data\\testing\\'
val_dir = 'data\\validation\\'

# model directory
model_dir = 'models/'
best_model_name = os.path.join(model_dir, 'best_model_%d.h5' % ITR)

# output directory
# results from show_result() will be save to here
output_dir = 'data\\result\\'

# variables for segmentation model training
BACKBONE = 'vgg16'
BATCH_SIZE = 2
CLASSES = ['joint']
LR = 0.0001  # learning rate
EPOCHS = 5

# define network parameters
n_classes = 1
activation = 'sigmoid'


def get_augmentation():
    """
    Get the augmentation steps. This is only used to resize images.
    :return: the augmentation steps as a callable object
    """

    transform = [
        album.Resize(height=320, width=320, always_apply=True),
    ]
    return album.Compose(transform)


def get_preprocess(fn=None):
    """
    Get the preprocessing steps.
    :return: the preprocessing steps as a callable object
    """

    # reference images directory (validation set)
    ref_dir = val_dir + 'XValidate\\'
    ref_images = [os.path.join(ref_dir, img) for img in os.listdir(ref_dir)]

    transform = [
        album.HistogramMatching(reference_images=ref_images, blend_ratio=(0.5, 0.5), p=1),
        # album.Equalize(p=1),
        album.CLAHE(clip_limit=1, p=1),
        # album.Lambda(fn),
    ]
    return album.Compose(transform)


def postprocess(image, smooth=False, largest_only=False, region_fill=False, resize_to=None):
    """
    Postprocess an image.

    :param image: the image to be processed
    :param smooth: whether to smoothen the edges
    :param largest_only: whether to keep only the largest connected component
    :param region_fill: whether to apply a contour filling
    :param resize_to: a shape tuple to resize image to, or None
    :return: the processed image
    """

    # remove salt and fill in holes
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(8, 8))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # smooth out the edges
    if smooth:
        image = cv2.pyrUp(image)
        for _ in range(5):
            image = cv2.medianBlur(image, 5)

    # keep the largest connected component
    if largest_only:
        image = keep_both_joints(image)

    # apply contour filling
    if region_fill:
        image = region_fill_mask(image)

    # resize to the given size
    if resize_to:
        image = cv2.resize(image, resize_to)

    # threshold the image into binary
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image
