import cv2
from image_stitching import ImageStitching



def forward():
    image_stitching = ImageStitching(query_image='inputs/desk/pa2.jpg',
                                 train_image='inputs/desk/pa1.jpg')
    _, train_photo_gray = image_stitching.read_images(image_stitching.train_image)
    _, query_photo_gray = image_stitching.read_images(image_stitching.query_image)
    
    keypoints_train_image, features_train_image = image_stitching._sift_detector(train_photo_gray) 
    keypoints_query_image, features_query_image = image_stitching._sift_detector(query_photo_gray) 
    
    matches = image_stitching.create_and_match_keypoints(features_train_image, features_query_image)
    
    M = image_stitching.compute_homography(keypoints_train_image, keypoints_query_image, matches, reprojThresh=4)
    
    if M is None: 
        print(f"Error")
        
    (matches, homography_matrix, status) = M
    
    result = image_stitching.blending_smoothing(homography_matrix)
    mapped_image = cv2.drawMatches(image_stitching.train_image, keypoints_train_image, image_stitching.query_image, keypoints_query_image, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return result, mapped_image