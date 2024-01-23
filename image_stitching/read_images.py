import cv2


def read(image_dir_list):
    """reads the images dir list and returns the images array as List

    Args:
        image_dir_list (List): image list read through cmdline

    Returns:
        images_list(List): List of numpy array of Images
        len(images_list) (int): no of images read through cmdline
    """
    images_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_list.append(image)

    return images_list, len(images_list)
