import panorama

def recurse(image_list, len_of_list):
    if len_of_list == 2:
        result = panorama.forward(query_photo=image_list[len_of_list-2], train_photo=image_list[len_of_list-1])
        
        return result
    else:
        result = panorama.forward(query_photo=image_list[len_of_list-2], train_photo=image_list[len_of_list-1])
        image_list[len_of_list-2] = result
        
        return recurse(image_list, len_of_list-1)
    
    