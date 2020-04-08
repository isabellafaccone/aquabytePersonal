import pandas as pd
from tqdm import tqdm
from random import randint, seed
from PIL import Image

LICE_CATEGORY = ['ADULT_FEMALE', 'MOVING', 'SCOTTISH_ADULT_FEMALE', 'UNSURE']


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


    

def generate_crops(lice_list, image_dim, crop_dim, categories = LICE_CATEGORY):
    iw, ih = image_dim
    cw, ch = crop_dim
    crops = {}
    for louse in lice_list:
        if louse['category'] not in categories:
            continue
        lp = louse['position'] 
        x, y, w, h = lp["left"], lp["top"], lp["width"], lp["height"]
        # append the louse to the crop that already covers it
        covered = False
        for crop in crops:
            if is_in_crop([x, y, w, h], list(crop) + crop_dim):
                crops[crop].append(louse)
                covered = True
        if not covered:
            crop_left_offset = randint(max(0, x + cw - iw), min(x, cw - w))
            crop_top_offset = randint(max(0, y + ch - ih), min(y, ch - h))
    
            crop_left = x - crop_left_offset
            crop_top = y - crop_top_offset
            crops[tuple([crop_left, crop_top])] = [louse]
    return crops
    

def is_in_crop(louse_xywh, crop_xywh):
    """Check if the bounding box of a lice falls inside the crop
    """
    x, y, w, h = louse_xywh
    crop_left, crop_top, crop_width, crop_height = crop_xywh
    
    crop_right, crop_bottom = crop_left + crop_width, crop_top + crop_height
    return x > crop_left and y > crop_top and x + w < crop_right and y + h < crop_bottom