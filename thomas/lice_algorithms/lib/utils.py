import csv

lice_labels_to_text_dict = {
    1: 'AFLice',
    2: 'OLice',
    3: 'Lice',
    -1: 'Not_AFLice',
    -2: 'Not_OLice'
}

def lice_labels_to_text(label):
    return lice_labels_to_text_dict[label]

def classify_lice_text(label_text):
    if 'Not_AFLice' in label_text:
        return -1
    elif 'Not_OLice' in label_text:
        return -2
    elif 'Not_Olice' in label_text:
        return -2
    elif 'AFLice' in label_text:
        return 1
    elif 'OLice' in label_text:
        return 2
    elif 'Olice' in label_text:
        return 2
    elif 'Lice' in label_text:
        return 3
    else:
        print label_text

def is_lice(label):
    return label > 0

def get_lice_annotations_from_file(annotations_file):
    f = open(annotations_file, 'rb')
    reader = csv.reader(f)

    lice_data = [ row[0].split() for row in reader]
    lice_data = [ (row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]), classify_lice_text(row[5])) for row in lice_data]

    return lice_data