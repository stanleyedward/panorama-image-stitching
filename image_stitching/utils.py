import cv2
from .image_stitching import ImageStitching
import sys
import numpy as np

def forward(query_photo, train_photo):
    """Runs a forward pass using the ImageStitching() class.
    Takes in a query image and train image and runs entire pipeline to return
    a panoramic image.
    
    Args:
        query_photo (numpy array): query image (RGB)
        train_photo (numpy array): train image (RGB)
        
    Returns:
        result_rgb (numpy array): RGB panorama image
        mapped_feature_image_rgb (numpy array): RGB visualization of matches
    """
    image_stitching = ImageStitching(query_photo, train_photo, type="cnn")
    
    _, query_photo_gray = image_stitching.give_gray(query_photo)
    _, train_photo_gray = image_stitching.give_gray(train_photo)
    
    
    if image_stitching.type == "cnn":
        keypoints_train_image, keypoints_query_image, matches = \
            image_stitching.extract_and_match_features(train_photo_gray, query_photo_gray)
    else: 
        keypoints_train_image, features_train_image = image_stitching._sift_detector(
            train_photo_gray
        )
        keypoints_query_image, features_query_image = image_stitching._sift_detector(
            query_photo_gray
        )

        matches = image_stitching.create_and_match_keypoints(
            features_train_image, features_query_image
        )
    
    print(f"Found {len(matches)} matches")

    mapped_feature_image = cv2.drawMatches(
        train_photo,
        keypoints_train_image,
        query_photo,
        keypoints_query_image,
        matches[:100],  # only get best 100 matches
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )
    
    if M is None:
        return "Error: cannot stitch images", None
    
    (matches, homography_matrix, status) = M
    
    result = image_stitching.blending_smoothing(
        query_photo, train_photo, homography_matrix
    )
    
    # Convert to proper format
    result_float32 = np.float32(result)
    mapped_float_32 = np.float32(mapped_feature_image)
    
    result_rgb = cv2.cvtColor(result_float32, cv2.COLOR_BGR2RGB)
    mapped_feature_image_rgb = cv2.cvtColor(mapped_float_32, cv2.COLOR_BGR2RGB)
    
    return result_rgb, mapped_feature_image_rgb


if __name__ == "__main__":
    try:
        query_image_path = sys.argv[1]
        train_image_path = sys.argv[2]
        
        def read_images(image_path):
            photo = cv2.imread(image_path)
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            return photo
        
        query_image = read_images(query_image_path)
        train_image = read_images(train_image_path)
        
        result, match_viz = forward(query_photo=query_image, train_photo=train_image)
        
        if isinstance(result, str):
            print(result)
        else:
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            match_viz_bgr = cv2.cvtColor(match_viz.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite("outputs/panorama_image.jpg", result_bgr)
            cv2.imwrite("outputs/matches_visualization.jpg", match_viz_bgr)
            print("Panorama saved to outputs/panorama_image.jpg")
            print("Matches saved to outputs/matches_visualization.jpg")
            
    except IndexError:
        print("Please input at least two source images")
        print("Usage: python script.py <query_image> <train_image>")