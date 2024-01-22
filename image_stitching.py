import numpy as np
import cv2

class ImageStitching():
    def __init__(self, query_image, train_image):
        super().__init__()
        self.query_image = query_image
        self.train_image = train_image
    
    def read_images(self, image):
        photo = cv2.imread(image)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        
        return photo, photo_gray
    
    @staticmethod
    def sift_detector(self, image):
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
        
        return keypoints, features
        
    def create_and_match_keypoints(self, features_train_image, features_query_image):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        best_matches = bf.match(features_train_image, features_query_image)
        raw_matches = sorted(best_matches, key= lambda x: x.distance)
        
        return raw_matches
    
    def compute_homography(self, keypoints_train_image, keypoints_query_image, matches, reprojThresh):
        keypoints_train_image = np.float32([keypoint.pt for keypoint in keypoints_train_image])
        keypoints_query_image = np.float32([keypoint.pt for keypoint in keypoints_query_image])
        
        if len(matches) >= 4:
            points_train = np.float32([keypoints_train_image[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_image[m.trainIdx] for m in matches])
            
            H, status = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
        
            return (matches, H, status)
    
        else: 
            print(f"Minimum match count not satisfied cannot get homopgrahy")
            return None