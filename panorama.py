import cv2
from image_stitching import ImageStitching

image_stitching = ImageStitching(query_image='inputs/desk/pa2.jpg',
                                 train_image='inputs/desk/pa1.jpg')

def forward():
    train_photo, train_photo_gray = image_stitching.read_images(image_stitching.train_image)
    query_photo, query_photo_gray = image_stitching.read_images(image_stitching.query_image)
    
    keypoints_train_image, features_train_image = image_stitching._sift_detector(train_photo_gray) 
    keypoints_query_image, features_query_image = image_stitching._sift_detector(query_photo_gray) 
    
    matches = image_stitching.create_and_match_keypoints(features_train_image, features_query_image)
    
    M = image_stitching.compute_homography(keypoints_train_image, keypoints_query_image, matches, reprojThresh=4)
    
    if M is None: 
        print(f"Error")