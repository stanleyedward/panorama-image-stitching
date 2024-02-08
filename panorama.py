import image_stitching.read_images as read_images
import image_stitching.recursion as recursion
import sys
import cv2


def main(image_dir_list):
    """Main function of the Repository.
    Takes in list of image dir, runs the complete image stitching pipeline
    to create and export a panoramic image in the /outputs/ folder.

    Args:
        image_dir_list (List): List of image dirs passed in cmdline
    """

    images_list, no_of_images = read_images.read(image_dir_list)
    result, mapped_image = recursion.recurse(images_list, no_of_images)
    cv2.imwrite("outputs/panorama_image.jpg", result)
    cv2.imwrite("outputs/mapped_image.jpg", mapped_image)

    print(f"Panoramic image saved at: outputs/panorama_image.jpg")


if __name__ == "__main__":
    image_list = []
    for i in range(1, len(sys.argv)):
        image_list.append(sys.argv[i])
    main(image_list)
