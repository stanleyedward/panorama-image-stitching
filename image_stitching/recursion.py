import image_stitching.utils as utils
import numpy as np
import cv2

def recurse(image_list, no_of_images):
    print(f"no of images: {no_of_images}")
    if no_of_images == 2:
        result = utils.forward(query_photo=image_list[no_of_images-2], train_photo=image_list[no_of_images-1])
        
        return result
    else:
        result = utils.forward(query_photo=image_list[no_of_images-2], train_photo=image_list[no_of_images-1])
        result_int8 = np.uint8(result)
        result_rgb = cv2.cvtColor(result_int8, cv2.COLOR_BGR2RGB)
        image_list[no_of_images-2] = result_rgb
        
        return recurse(image_list, no_of_images-1)
    
    