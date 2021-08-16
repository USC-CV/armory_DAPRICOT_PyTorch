from typing import Callable

import tensorflow as tf
import numpy as np
import cv2
from armory.data import datasets
from armory.data.adversarial import (  # noqa: F401
    imagenet_adversarial as IA,
    librispeech_adversarial as LA,
    resisc45_densenet121_univpatch_and_univperturbation_adversarial_224x224,
    ucf101_mars_perturbation_and_patch_adversarial_112x112,
    gtsrb_bh_poison_micronnet,
    apricot_dev,
    apricot_test,
    dapricot_dev,
    dapricot_test,
)

dapricot_adversarial_context = datasets.ImageContext(x_shape=(3, None, None, 3))


def ava_dapricot_canonical_preprocessing(batch):
    # DAPRICOT raw images are rotated by 90 deg and color channels are BGR, so the
    # following line corrects for this
    batch_rotated_rgb = np.transpose(batch, (0, 1, 3, 2, 4))[:, :, :, ::-1, :]

    # Ava resize images
    new_batch_rotated_rgb = []
    for img in batch_rotated_rgb[0]:
        h, w, c = img.shape
        resize_img = cv2.resize(img, (w//3, h//3))
        new_batch_rotated_rgb.append(resize_img)
        break
    new_batch_rotated_rgb = np.stack(new_batch_rotated_rgb)
    new_batch_rotated_rgb = np.expand_dims(new_batch_rotated_rgb, axis=0)

    return datasets.canonical_variable_image_preprocess(
        dapricot_adversarial_context, new_batch_rotated_rgb
    )



def ava_dapricot_label_preprocessing(x, y):
    """
    """
    y_object, y_patch_metadata = y
    y_object_list = []
    y_patch_metadata_list = []
    # each example contains images from N cameras, i.e. N=3
    num_imgs_per_ex = np.array(y_object["id"].flat_values).size
    y_patch_metadata["gs_coords"] = np.array(
        # Ava resize images
        y_patch_metadata["gs_coords"].flat_values/3
    ).reshape((num_imgs_per_ex, -1, 2))
    y_patch_metadata["shape"] = y_patch_metadata["shape"].reshape((num_imgs_per_ex,))
    y_patch_metadata["cc_scene"] = y_patch_metadata["cc_scene"][0]
    y_patch_metadata["cc_ground_truth"] = y_patch_metadata["cc_ground_truth"][0]
    for i in range(num_imgs_per_ex):
        y_object_img = {}
        for k, v in y_object.items():
            y_object_img[k] = np.array(y_object[k].flat_values[i])
            # Ava resize images
            if k == "area":
                y_object_img[k] = np.array(y_object[k].flat_values[i]/9)
        y_object_list.append(y_object_img)

        y_patch_metadata_img = {
            k: np.array(y_patch_metadata[k][i]) for k, v in y_patch_metadata.items()
        }
        y_patch_metadata_list.append(y_patch_metadata_img)
        break

    return (y_object_list, y_patch_metadata_list)