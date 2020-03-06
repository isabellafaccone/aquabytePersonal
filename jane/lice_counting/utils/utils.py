from PIL import Image

    
def xywh2yolobbox(xywh, image_dim):   
    """Convert bouding box from xywh to normalized format for yolo model
    
    Parameters: 
    ----------
    xywh : list
        bounding box of top-left corner x, y with width, height
    image_dim : list
        image [width, height]
  
    Returns: 
    list: 
        center of rectangle with width and height relative to width and height of image (0.0 to 1.0]
    """
    x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    iw, ih = image_dim[0], image_dim[1]
    x_center = x + w / 2
    y_center = y + h / 2
    return [x_center / iw, y_center / ih, w / iw, h / ih]


def write_label(class_index, bbox, file_name, path):
    f = open("{}/{}.txt".format(path, file_name), "w+")
    bbox_str = " ".join([str(i) for i in bbox])
    f.write("{} {}".format(class_index, bbox_str))
    f.close() 

    
def write_image(image, file_name, path):
    img = Image.fromarray(image , 'RGB')
    img.save("{}/{}.jpg".format(path, file_name), 'JPEG')


    
  