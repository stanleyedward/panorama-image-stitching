import cv2
from image_stitching import ImageStitching
import sys

def forward(query_image_dir, train_image_dir):
    image_stitching = ImageStitching()
    query_photo, query_photo_gray = image_stitching.read_images('inputs/desk/pa1.jpg') #left image
    train_photo, train_photo_gray = image_stitching.read_images('inputs/desk/pa2.jpg') #right image
    
    
    keypoints_train_image, features_train_image = image_stitching._sift_detector(train_photo_gray) 
    keypoints_query_image, features_query_image = image_stitching._sift_detector(query_photo_gray) 
    
    matches = image_stitching.create_and_match_keypoints(features_train_image, features_query_image)
    
    M = image_stitching.compute_homography(keypoints_train_image, keypoints_query_image, matches, reprojThresh=4)
    
    if M is None: 
        print(f"Error")
        
    (matches, homography_matrix, status) = M
    
    result = image_stitching.blending_smoothing(query_photo, train_photo, homography_matrix)
    mapped_image = cv2.drawMatches(train_photo, keypoints_train_image, query_photo, keypoints_query_image, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # cv2.imshow("keypoint matches", mapped_image)
    # cv2.imshow("panorama", result)
    
    cv2.imwrite("outputs/matched_points.jpg", mapped_image)
    cv2.imwrite("outputs/panorama_image.jpg", result)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__=='__main__':
    try:
        # forward(query_image=sys.argv[1], train_image= sys.argv[2])
        forward()
    except IndexError:
        print ("Please input atleast two source images")
        