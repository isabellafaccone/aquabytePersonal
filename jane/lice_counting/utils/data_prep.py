import pandas as pd
from tqdm import tqdm
from random import randint, seed
from PIL import Image

LICE_CATEGORY = ['ADULT_FEMALE', 'MOVING']


def write_label(class_index, bbox, file_name, path):
    f = open("{}/{}.txt".format(path, file_name), "w+")
    bbox_str = " ".join([str(i) for i in bbox])
    f.write("{} {}".format(class_index, bbox_str))
    f.close() 

    
def write_labels(lice_labels, file_name, path):
    f = open("{}/{}.txt".format(path, file_name), "w+")
    for label in lice_labels:
        class_index, bbox = label[0], label[1:]
        bbox_str = " ".join([str(i) for i in bbox])
        f.write("{} {}\n".format(class_index, bbox_str))
    f.close() 
    
def write_image(image, file_name, path):
    img = Image.fromarray(image , 'RGB')
    img.save("{}/{}.jpg".format(path, file_name), 'JPEG')

        
        
        
    
def get_df_ad(df): 
    """Filter fish crop data that have adult female lice 

    Parameters: 
    ----------
    df : pandas.DataFrame
        with column "annotation"
  
    Returns: 
    pandas.DataFrame: 
        Filtered rows in df that have ADULT_FEMALE category in annotation
    """
    
    new_df = pd.DataFrame(columns = list(df.columns.values))
    for idx, row in tqdm(df.iterrows()):
        if row['annotation'] is not None:
            new_lice_annotation = []
            for lice in row['annotation']:
                if lice['category'] == 'ADULT_FEMALE':
                    new_lice_annotation.append(lice)
            if len(new_lice_annotation) > 0:
                row['annotation'] = new_lice_annotation
                new_df = new_df.append(row)
    return new_df


def generate_crop_smart(lice, image_dim, crop_dim):
    iw, ih = image_dim
    cw, ch = crop_dim

    lp = lice['position'] 
    x, y, w, h = lp["left"], lp["top"], lp["width"], lp["height"]    
    top_buffer = max(100, y + ch - ih)
    bottom_buffer = max(150, ch - h - y)
    

    left_buffer = min(int(cw / 4), cw - w, x)
    right_buffer = max(int(cw * 3 / 4), x + cw - iw)

    
    left_offset_min = max(0, x + cw - iw)
    left_offset_max = min(x, cw - w)
    
    top_offset_min = max(0, y + ch - ih)
    top_offset_max = min(y, ch - h)
    
    if lice['location'] == "TOP":
        top_offset_max = min(top_offset_max, top_buffer)
    elif lice['location'] == "BOTTOM":
        top_offset_min = max(top_offset_min, ch - h - bottom_buffer)
    else:
        left_offset_min = max(left_offset_min, left_buffer)
        left_offset_max = min(left_offset_max, right_buffer)
        
    crop_left_offset = randint(left_offset_min, left_offset_max)
    crop_top_offset = randint(top_offset_min, top_offset_max)
    crop_left = x - crop_left_offset
    crop_top = y - crop_top_offset
    return crop_left, crop_top


def generate_crops_smart(lice_list, image_dim, crop_dim, categories = LICE_CATEGORY):
    """
    Return:
    Dictionary key: crop left and top, value: list of lice covered
    """
    crops = {}
    for lice in lice_list:
        if lice['category'] not in categories:
            continue
        covered = False
        lp = lice['position'] 
        x, y, w, h = lp["left"], lp["top"], lp["width"], lp["height"]
        for crop in crops:
            if is_in_crop([x, y, w, h], list(crop) + crop_dim):
                crops[crop].append(lice)
                covered = True
        if not covered:
            crop_left, crop_top = generate_crop_smart(lice, image_dim, crop_dim)
            crops[tuple([crop_left, crop_top])] = [lice]   
    return crops


def generate_lice_cluster(lice_list, crop_dim, categories = LICE_CATEGORY):
    """
    Return:
    Dictionary key: crop left and top, value: list of lice covered
    """
    cluster = {}
    for lice in lice_list:
        if lice['category'] not in categories:
            continue
        covered = False
        lp = lice['position'] 
        x, y, w, h = lp["left"], lp["top"], lp["width"], lp["height"]
        for crop in crops:
            if is_in_crop([x, y, w, h], list(crop) + crop_dim):
                crops[crop].append(lice)
                covered = True
        if not covered:
            crop_left, crop_top = generate_crop_smart(lice, image_dim, crop_dim)
            crops[tuple([crop_left, crop_top])] = [lice]   
    return crops


def generate_crops_uniform(lice_list, image_dim, crop_dim, categories = LICE_CATEGORY):
    iw, ih = image_dim
    cw, ch = crop_dim
    crops = {}
    for lice in lice_list:
        if lice['category'] not in categories:
            continue
        lp = lice['position'] 
        x, y, w, h = lp["left"], lp["top"], lp["width"], lp["height"]
        # append the lice to the crop that already covers it
        covered = False
        for crop in crops:
            if is_in_crop([x, y, w, h], list(crop) + crop_dim):
                crops[crop].append(lice)
                covered = True
        if not covered:
            crop_left_offset = randint(max(0, x + cw - iw), min(x, cw - w))
            crop_top_offset = randint(max(0, y + ch - ih), min(y, ch - h))
    
            crop_left = x - crop_left_offset
            crop_top = y - crop_top_offset
            crops[tuple([crop_left, crop_top])] = [lice]
    return crops
    

def is_in_crop(lice_xywh, crop_xywh):
    """Check if the bounding box of a lice falls inside the crop
    """
    x, y, w, h = lice_xywh
    crop_left, crop_top, crop_width, crop_height = crop_xywh
    
    crop_right, crop_bottom = crop_left + crop_width, crop_top + crop_height
    return x > crop_left and y > crop_top and x + w < crop_right and y + h < crop_bottom