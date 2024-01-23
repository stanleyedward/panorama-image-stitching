import cv2
from image_stitching import ImageStitching
import sys
import numpy as np


def forward(query_image_dir, train_image_dir):
    image_stitching = ImageStitching()
    query_photo, query_photo_gray = image_stitching.read_images(
        query_image_dir
    )  # left image
    train_photo, train_photo_gray = image_stitching.read_images(
        train_image_dir
    )  # right image

    keypoints_train_image, features_train_image = image_stitching._sift_detector(
        train_photo_gray
    )
    keypoints_query_image, features_query_image = image_stitching._sift_detector(
        query_photo_gray
    )

    matches = image_stitching.create_and_match_keypoints(
        features_train_image, features_query_image
    )

    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        print(f"Error")

    (matches, homography_matrix, status) = M

    result = image_stitching.blending_smoothing(
        query_photo, train_photo, homography_matrix
    )
    # mapped_image = cv2.drawMatches(train_photo, keypoints_train_image, query_photo, keypoints_query_image, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    result_float32 = np.float32(result)
    result_rgb = cv2.cvtColor(result_float32, cv2.COLOR_BGR2RGB)
    
    return result_rgb


if __name__ == "__main__":
    try:
        result = forward(query_image_dir=sys.argv[1], train_image_dir=sys.argv[2])
        cv2.imwrite("outputs/panorama_image.jpg", result)
    except IndexError:
        print("Please input atleast two source images")
