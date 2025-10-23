import numpy as np
import cv2
from lightglue import LightGlue, DISK
from lightglue.utils import rbd
import torch

class ImageStitching:
    """contains the utilities required to stitch images"""

    def __init__(self, query_photo, train_photo, type="cnn"):
        super().__init__()
        width_query_photo = query_photo.shape[1]
        width_train_photo = train_photo.shape[1]
        lowest_width = min(width_query_photo, width_train_photo)
        smoothing_window_percent = 0.10
        self.smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))
        self.type = type
        if type == "cnn":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='disk').eval().to(self.device)
            print(f"[INFO]loaded both disk and lightglue models!")

    def give_gray(self, image):
        """receives an image array and returns grayscaled image

        Args:
            image (numpy array): array of images

        Returns:
            image (numpy array): same as image input
            photo_gray (numpy array): grayscaled images
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            photo_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            photo_gray = image
        return image, photo_gray

    @staticmethod
    def _sift_detector(image):
        """Applies SIFT algorithm to the given image

        Args:
            image (numpy array): input image

        Returns:
            keypoints, features
        """
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)

        return keypoints, features

    def create_and_match_keypoints(self, features_train_image, features_query_image):
        """Creates and Matches keypoints from the SIFT features using Brute Force matching
        by checking the L2 norm of the feature vector

        Args:
            features_train_image: SIFT features of train image
            features_query_image: SIFT features of query image

        Returns:
            matches (List): matches in features of train and query image
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        best_matches = bf.match(features_train_image, features_query_image)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)

        return raw_matches
    
    def _prepare_image_for_disk(self, gray_image):
        """Convert grayscale image to format expected by DISK
        
        Args:
            gray_image: numpy array HxW grayscale image
            
        Returns:
            torch tensor: 1x3xHxW normalized to [0,1]
        """
        img_3ch = np.stack([gray_image] * 3, axis=0)  # [3, H, W]
        img_tensor = torch.from_numpy(img_3ch / 255.0).float().unsqueeze(0).to(self.device)
        return img_tensor

    def extract_and_match_features(self, train_gray, query_gray):
        """Extract DISK features and match using LightGlue
        
        Args:
            train_gray: grayscale train image (numpy array)
            query_gray: grayscale query image (numpy array)
            
        Returns:
            keypoints_train: list of cv2.KeyPoint
            keypoints_query: list of cv2.KeyPoint
            matches: list of cv2.DMatch
        """
        image0 = self._prepare_image_for_disk(train_gray)
        image1 = self._prepare_image_for_disk(query_gray)
        
        with torch.inference_mode():
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)
            
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            
            match_indices = matches01['matches']  # shape (K, 2)
            match_scores = matches01['scores']  # shape (K,)
            
            kpts0 = feats0['keypoints'].cpu().numpy()  # shape (N0, 2)
            kpts1 = feats1['keypoints'].cpu().numpy()  # shape (N1, 2)
        
        keypoints_train = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in kpts0]
        keypoints_query = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0) for pt in kpts1]
        
        matches = []
        match_indices_np = match_indices.cpu().numpy()
        match_scores_np = match_scores.cpu().numpy()
        
        for i in range(len(match_indices_np)):
            idx0 = int(match_indices_np[i, 0])  # index in image0 (train)
            idx1 = int(match_indices_np[i, 1])  # index in image1y(query)
            score = float(match_scores_np[i])
            
            distance = 1.0 - score
            match = cv2.DMatch(idx0, idx1, distance)
            matches.append(match)
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        print(f"DISK+LightGlue: Extracted {len(keypoints_train)} train keypoints, "
              f"{len(keypoints_query)} query keypoints, found {len(matches)} matches")
        
        return keypoints_train, keypoints_query, matches

    def compute_homography(
        self, keypoints_train_image, keypoints_query_image, matches, reprojThresh
    ):
        """Computes the Homography to map images to a single plane,
        uses RANSAC algorithm to find the best matches iteratively.

        Args:
            keypoints_train_image: keypoints found in train image
            keypoints_query_image: keypoints found in query image
            matches: matches found using LightGlue
            reprojThresh: threshold for error

        Returns:
            M (Tuple): (matches, Homography matrix, status)
        """
        keypoints_train_image = np.float32(
            [keypoint.pt for keypoint in keypoints_train_image]
        )
        keypoints_query_image = np.float32(
            [keypoint.pt for keypoint in keypoints_query_image]
        )

        if len(matches) >= 4:
            points_train = np.float32(
                [keypoints_train_image[m.queryIdx] for m in matches]
            )
            points_query = np.float32(
                [keypoints_query_image[m.trainIdx] for m in matches]
            )

            H, status = cv2.findHomography(
                points_train, points_query, cv2.RANSAC, reprojThresh
            )

            return (matches, H, status)
        else:
            print(f"Minimum match count not satisfied, cannot get homography")
            return None

    def create_mask(self, query_image, train_image, version):
        """Creates the mask using query and train images for blending the images,
        using a gaussian smoothing window/kernel

        Args:
            query_image (numpy array)
            train_image (numpy array)
            version (str) == 'left_image' or 'right_image'

        Returns:
            masks
        """
        height_query_photo = query_image.shape[0]
        width_query_photo = query_image.shape[1]
        width_train_photo = train_image.shape[1]
        height_panorama = height_query_photo
        width_panorama = width_query_photo + width_train_photo
        offset = int(self.smoothing_window_size / 2)
        barrier = query_image.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == "left_image":
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset :] = 1
        return cv2.merge([mask, mask, mask])

    def blending_smoothing(self, query_image, train_image, homography_matrix):
        """blends both query and train image via the homography matrix,
        and ensures proper blending and smoothing using masks created in create_masks()
        to give a seamless panorama.

        Args:
            query_image (numpy array)
            train_image (numpy array)
            homography_matrix (numpy array): Homography to map images to a single plane

        Returns:
            panoramic image (numpy array)
        """
        height_img1 = query_image.shape[0]
        width_img1 = query_image.shape[1]
        width_img2 = train_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(query_image, train_image, version="left_image")
        panorama1[0 : query_image.shape[0], 0 : query_image.shape[1], :] = query_image
        panorama1 *= mask1
        mask2 = self.create_mask(query_image, train_image, version="right_image")
        panorama2 = (
            cv2.warpPerspective(
                train_image, homography_matrix, (width_panorama, height_panorama)
            )
            * mask2
        )
        result = panorama1 + panorama2

        # remove extra blackspace
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result