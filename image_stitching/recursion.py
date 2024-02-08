import image_stitching.utils as utils
import numpy as np
import cv2


def recurse(image_list, no_of_images):
    """Recursive function to get panorama of multiple images

    Args:
        image_list (List): list of numpy array of images
        no_of_images (int): no of images read through cmdline

    Returns:
        result (numpy array): RGB panoramic image
    """
    if no_of_images == 2:
        result, mapped_image = utils.forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )

        return result, mapped_image
    else:
        result,_ = utils.forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )
        result_int8 = np.uint8(result)
        result_rgb = cv2.cvtColor(result_int8, cv2.COLOR_BGR2RGB)
        image_list[no_of_images - 2] = result_rgb

        return recurse(image_list, no_of_images - 1)
