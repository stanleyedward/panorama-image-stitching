import cv2
from .image_stitching import ImageStitching
import sys
import numpy as np


def forward(query_photo, train_photo):
    """Runs a forward pass using the ImageStitching() class in utils.py.
    Takes in a query image and train image and runs entire pipeline to return
    a panoramic image.

    Args:
        query_photo (numpy array): query image
        train_photo (nnumpy array): train image

    Returns:
        result image (numpy array): RGB result image
    """
    image_stitching = ImageStitching(query_photo, train_photo)
    _, query_photo_gray = image_stitching.give_gray(query_photo)  # left image
    _, train_photo_gray = image_stitching.give_gray(train_photo)  # right image

    keypoints_train_image, features_train_image = image_stitching._sift_detector(
        train_photo_gray
    )
    keypoints_query_image, features_query_image = image_stitching._sift_detector(
        query_photo_gray
    )

    matches = image_stitching.create_and_match_keypoints(
        features_train_image, features_query_image
    )

    mapped_feature_image = cv2.drawMatches(
                        train_photo,
                        keypoints_train_image,
                        query_photo,
                        keypoints_query_image,
                        matches[:100],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        return "Error cannot stitch images"

    (matches, homography_matrix, status) = M

    result = image_stitching.blending_smoothing(
        query_photo, train_photo, homography_matrix
    )
    # mapped_image = cv2.drawMatches(train_photo, keypoints_train_image, query_photo, keypoints_query_image, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    mapped_float_32 = np.float32(mapped_feature_image)
    result_float32 = np.float32(result)
    result_rgb = cv2.cvtColor(result_float32, cv2.COLOR_BGR2RGB)
    mapped_feature_image_rgb = cv2.cvtColor(mapped_float_32, cv2.COLOR_BGR2RGB)
    
    return result_rgb, mapped_feature_image_rgb


if __name__ == "__main__":
    try:
        query_image = sys.argv[1]
        train_image = sys.argv[2]

        def read_images(image):
            photo = cv2.imread(image)
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

            return photo

        query_image = read_images(query_image)
        train_image = read_images(train_image)

        result = forward(query_photo=query_image, train_photo=train_image)
        cv2.imwrite("outputs/panorama_image.jpg", result)
    except IndexError:
        print("Please input atleast two source images")
