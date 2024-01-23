import image_stitching.utils as utils

def recurse(image_list, no_of_images):
    if no_of_images == 2:
        result = utils.forward(query_photo=image_list[no_of_images-2], train_photo=image_list[no_of_images-1])
        
        return result
    else:
        result = utils.forward(query_photo=image_list[no_of_images-2], train_photo=image_list[no_of_images-1])
        image_list[no_of_images-2] = result
        
        return recurse(image_list, no_of_images-1)
    
    