import logging

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def hungarian_matcher(left_ids, left_bottom_top_edge_locations, right_ids, right_bottom_top_edge_locations):
    """
    TBD
    Returns
        a list of left and right id pair. If either id is None, it is an unmatched item.
    """
    # match the bboxes. Return a list of matched bboxes
    COST_THRESHOLD = 100.0

    pairs = []
    if left_ids and right_ids:
        # pairwise euclidean distance matrix
        cost_matrix = cdist(left_bottom_top_edge_locations, right_bottom_top_edge_locations, metric='euclidean')

        # hungarian algorithm to minimize weights in bipartite graph
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # move matched items from left_ids/right_ids to pairs
        for (r, c) in zip(row_ind, col_ind):
            if cost_matrix[r, c] < COST_THRESHOLD:
                pairs.append((left_ids[r], right_ids[c]))
                left_ids[r] = None
                right_ids[c] = None

    # unmatched singles
    lefts = [(key, None) for key in left_ids if key]
    rights = [(None, key) for key in right_ids if key]

    logging.info("hungarian_matcher left={}, right={} -> matched={}, left={}, right={}".format(
        len(left_ids), len(right_ids), len(pairs), len(lefts), len(rights)))

    # merge all into pairs as final result
    pairs.extend(lefts)
    pairs.extend(rights)
    return pairs