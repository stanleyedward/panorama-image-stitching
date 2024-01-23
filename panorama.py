import image_stitching.read_images as read_images
import image_stitching.recursion as recursion
import sys
import cv2

def main(image_dir_list):
    images_list, no_of_images = read_images.read(image_dir_list)
    result = recursion.recurse(images_list, no_of_images)
    cv2.imwrite("outputs/panorama_image.jpg", result)
    
if __name__=='__main__':
    main([sys.argv[1], sys.argv[2]])