import os
from random import choice
from shutil import copy
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def vis(**images):
    """
    Plot images side by side with provided titles and np.ndarray's.

    :param images:  a dictionary containing elements title=image
    e.g.,
        arg = { 'image': img, 'mask': mask }
        vis(**arg)
    which is equivalent to
        vis(image=img, mask=mask)
    """

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name).title())
        plt.imshow(image)
    plt.show()


class Dataset:
    """
    Class for data loading and preprocessing.
    """

    # class names
    CLASSES = ['joint']

    def __init__(self, images_dir: str, masks_dir: str,
                 classes=None, aug=None, prep=None, img_ext='png'):
        """
        Constructs a dataset.

        :param images_dir: directory of the images
        :param masks_dir: directory of the masks
        :param classes: classes to detect, should be one or more of those in CLASSES
        :param aug: augmentation function
        :param prep: preprocessing function
        :param img_ext: image extension; 'png' by default
        """

        # set up class variables
        self.ids = [i for i in sorted(os.listdir(images_dir)) if i.endswith(img_ext)]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))
                          if image_id.endswith(img_ext)]

        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.aug = aug
        self.prep = prep

    def __getitem__(self, i):
        """
        Gets the ith item in this dataset.
        Usage: dataset[i]

        :param i: the index
        :return: the ith preprocessed sample (image, mask) as np.ndarray
        """

        # read data
        image = self.get_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.get_mask(i)
        mask = cv2.bitwise_not(mask)

        # brighten dark pixels
        # this may not be needed for all kinds of data
        image[image <= 50] += 20

        # extract classes from mask
        masks = [(mask == v) for v in self.classes]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.prep:
            sample = self.prep(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def shapeof(self, i):
        """
        Get the shape of the ith sample.

        :param i: the index
        :return: the shape as a tuple
        """
        return cv2.imread(self.images_fps[i]).shape

    def idof(self, i):
        """
        Get the file name of the ith image.

        :param i: the index
        :return: the file name as a string
        """
        return self.ids[i]

    def get_image(self, i):
        """
        Get the ith image.

        :param i: the index
        :return: the ith image as an np.ndarray
        """
        return cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)

    def get_mask(self, index):
        """
        Get the ith mask.

        :param index: the index
        :return: the ith mask as an np.ndarray
        """
        return cv2.imread(self.masks_fps[index], cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        """
        Get the length of this dataset.

        :return: the number of samples as an int
        """
        return len(self.ids)


class Dataloader(tf.keras.utils.Sequence):
    """
    A Keras Sequence class for more convenient data loading for the model.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Load data from dataset and form batches.

        :param dataset: instance of Dataset class for image loading and preprocessing
        :param batch_size: integer number of images in batch
        :param shuffle: boolean; if True, shuffle indices for each epoch
        """

        # initialize class variables
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

        # call on_epoch_end() to shuffle data if needed
        self.on_epoch_end()

    def on_epoch_end(self):
        """Callback function to shuffle indices for each epoch"""
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)

    def __getitem__(self, i):
        """
        Get the ith batch.

        :param i: the index
        :return: the ith batch as a tuple
        """

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for i in range(start, stop):
            data.append(self.dataset[i])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        """
        :returns the number of batches in this loader
        """
        return len(self.indices) // self.batch_size


def graph_loss(history):
    """
    Graph the training and validation loss of a model.

    :param history: the history object returned by model.fit
    """

    plt.figure(figsize=(30, 5))
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def print_metrics(scores: list, metrics: list):
    """
    Print the metrics from a model evaluation.

    :param scores: list of scalars returned by model.evaluate()
    :param metrics: list of metrics
    """

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))


def delete_all(path: str):
    """
    Delete all files in a directory.
    :param path: the directory path
    :return:
    """
    files = os.listdir(path)
    for f in files:
        os.remove(os.path.join(path, f))


def show_results(dataset: Dataset, model, n: int, postprocess=None,
                 show=True, save=False, out='', **post_args):
    """
    Show and process results obtained from predicting samples in dataset
    using model.

    :param dataset: a Dataset object
    :param model: a model
    :param n: number of samples to process
    :param postprocess:
    :param show: graph the images and masks if True
    :param save: save the images to out if True
    :param out: output directory
    :param post_args: additional postprocessing arguments
    """

    # delete all original files in the output directory
    if save and os.path.exists(out):
        delete_all(out)

    for i in range(n):
        # get a test image
        image, mask = dataset[i]
        joint_id = dataset.idof(i)
        image = np.expand_dims(image, axis=0)

        # get the predicted mask
        pred_mask = model.predict(image).round()

        # postprocessing
        image = image.squeeze()
        mask = mask[..., 0].squeeze()
        pred_mask = pred_mask[..., 0].squeeze()

        # make the predicted mask [0..255]
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask *= 255

        # get the original shape of this sample
        og_shape = dataset.shapeof(i)
        post_args['resize_to'] = (og_shape[1], og_shape[0])

        # visualization arguments
        vis_arg = {
            joint_id: dataset.get_image(i),
            'preprocessed': image,
            'original_mask': mask,
            'predicted_mask': pred_mask,
        }

        # postprocess result
        post = None
        if postprocess:
            post = postprocess(image=pred_mask, **post_args)
            vis_arg['postprocessed'] = post

        # display result
        if show:
            vis(**vis_arg)

        # save the resulting image
        if save:
            # create folder if output directory does not exist
            if not os.path.exists(out):
                os.mkdir(out)

            filename = joint_id[:joint_id.find('.')] + '_pred.png'
            status = cv2.imwrite(out + filename, post if postprocess else pred_mask)
            if not show:
                from IPython.display import clear_output
                clear_output(wait=True)

            print(f'Image #{i+1} ({filename}) written: {status}')


def save_normalize(img_from: str, img_to: str) -> bool:
    """
    Turn all non-black pixels in a mask to white pixels and save it.

    :param img_from: source file path
    :param img_to: destination file path
    :return: cv2.imwrite() status as a bool
    """
    img = cv2.imread(img_from)
    img[img != 0] = 255

    return cv2.imwrite(img_to, img)

def keep_both_joints(mask: np.ndarray, mark=122) -> np.ndarray:
    if len(mask.shape) != 2:
        raise ValueError('Image must be binary')

    height, width = mask.shape

    point1 = (int(width / 2), int(height / 8))
    np_point_1 = (point1[1], point1[0])

    point2 = (int(width / 2), int(7 * height / 8))
    np_point_2 = (point2[1], point2[0])

    while point1[1] < height-1 and mask[np_point_1] == 0:
        point1 = (point1[0], point1[1] + 1)
        np_point_2 = (point1[1], point1[0])

    while point2[1] < height-1 and mask[np_point_2] == 0:
        point2 = (point2[0], point2[1] - 1)
        np_point_2 = (point2[1], point2[0])

    cv2.floodFill(mask, None, seedPoint=point1, newVal=mark)
    cv2.floodFill(mask, None, seedPoint=point2, newVal=mark)
    mask[mask != mark] = 0    # non-largest component pixel becomes black
    mask[mask == mark] = 255  # largest component becomes white

    return mask

def keep_largest_comp(mask: np.ndarray, mark=122) -> np.ndarray:
    """
    Naively search for the largest component starting from the center diagonally,
    then keep only the connected component of the first white pixel found.

    :param mask: the mask
    :param mark: the temporary pixel intensity
    :return: the processed mask
    """

    if len(mask.shape) != 2:
        raise ValueError('Image must be binary')

    height, width = mask.shape
    point = (int(width / 2), int(height / 2))   # (x, y) for cv2
    np_point = (point[1], point[0])             # (y, x) for numpy

    # in case the center pixel is not part of the hand
    # go down until a white pixel is found
    while point[0] < width-1 and point[1] < height-1 and mask[np_point] == 0:
        point = (point[0] + 1, point[1] + 1)
        np_point = (point[1], point[0])

    cv2.floodFill(mask, None, seedPoint=point, newVal=mark)
    mask[mask != mark] = 0    # non-largest component pixel becomes black
    mask[mask == mark] = 255  # largest component becomes white

    return mask


def strip_joint_coord(txt: str) -> np.ndarray:
    """
    Process through the OA text file and get the ground-truth joint
    coordinates.

    :param txt: the text file
    :return: the joint coordinates as an numpy array of shape (12, 2)
    """

    points = np.zeros(shape=(12, 2), dtype=np.uint)
    idx = 0
    with open(txt) as f:
        f.readline()  # skip first line

        for line in f:
            tokens = line.split()
            if len(tokens) != 4:
                raise ValueError("Invalid format: " + line)

            # cv2 needs x and y reversed
            points[idx][0] = int(tokens[2])
            points[idx][1] = int(tokens[1])
            idx += 1

    return points


def joint_accuracy(mask, points) -> float:
    """
    Calculates and returns the accuracy of the mask.
    Accuracy is calculated by the number the joints the predicted mask successfully overlaps.
    Locations of the joints are obtained from the QA .txt files.

    :param mask:    the mask
    :param points:  the coordinates of 12 joints
    :return:        the number ofhand_accuracy coordinates the mask overlaps / the total
    """
    if type(mask) is str:
        mask = cv2.imread(mask, cv2.THRESH_BINARY)
    elif not type(mask) is np.ndarray:
        raise ValueError('mask type not supported: ' + str(type(mask)))

    pixels = np.zeros(shape=(12, 1), dtype=np.uint)
    idx = 0
    for p in points:
        pixels[idx][0] = mask[p[0], p[1]] // 255
        idx += 1

    return np.sum(pixels) / 12


def color_joints(img, joints, color=None, size=10):
    """
    Color on the image the joints provided by the joint coordinates.

    :param img: the image
    :param joints: the joint coordinates
    :param color: the color; red by default
    :param size: the size of the mark
    :return: the processed image
    """

    if type(img) is str:
        img = cv2.imread(img)

    if color is None:
        color = [0, 0, 255]  # red

    # mark the joint of the image by coloring a square
    for joint in joints:
        i = joint[0]
        j = joint[1]
        img[i - size:i + size + 1, j - size:j + size + 1] = color

    return img


def mask_diff(mask1, mask2) -> int:
    """
    Compare two masks and return the number of difference pixels.

    :param mask1: the first mask
    :param mask2: the second mask
    :return: the number of different pixels
    """
    if type(mask1) is str:
        mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
    if type(mask2) is str:
        mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)

    assert type(mask1) is np.ndarray and type(mask2) is np.ndarray
    diff = mask1 - mask2
    return len(diff[diff != 0])


def region_fill_mask(mask):
    """
    Apply a region fill on the mask.

    :param mask: the mask
    :return: the processed mask
    """
    result = np.copy(mask)
    contour, hier = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in contour:
        result = cv2.drawContours(result, [i], 0, 255, -1)

    return result
