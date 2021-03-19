from hungarian_matcher import hungarian_matcher

BATI = "BATI"
LATI = "LATI"

def match_annotations(crop_annotations, service):
    c_s = crop_annotations.select(service=service)

    left_bottom_top_edge_locations = []
    left_ids = []
    for crop in c_s.left:
        left_ids.append(crop.id)
        bbox = crop.bbox
        left_bottom_top_edge_locations.append([bbox[2], bbox[0]])

    right_bottom_top_edge_locations = []
    right_ids = []
    for crop in c_s.right:
        right_ids.append(crop.id)
        bbox = crop.bbox
        right_bottom_top_edge_locations.append([bbox[2], bbox[0]])

    pairs = hungarian_matcher(left_ids, left_bottom_top_edge_locations, right_ids, right_bottom_top_edge_locations)

    # assign pair_id to the matched (and for LATI, also unmatched)
    for left_id, right_id in pairs:
        if (service == BATI) and not (left_id and right_id):
            continue
        pair_id = crop_annotations.get_pair_id(left_id, right_id)
        if left_id:
            crop_annotations[left_id].pair_id = pair_id
        if right_id:
            crop_annotations[right_id].pair_id = pair_id
    return pairs

