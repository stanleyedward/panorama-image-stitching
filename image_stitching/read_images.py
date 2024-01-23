import cv2


def read(image_dir_list):
    images_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_list.append(image)

    return images_list, len(images_list)
