import numpy as np
from utils import convert_to_world_point
from sklearn import decomposition

def get_bb_from_mask(mask):
    """
    Computes the bounding box coordinates from the mask

    Input:
        - mask : np.array of size (L, H)

    Output:
        - (x1, x2, y1, y2) : coordinates of the corner of
        the bounding box
    """
    x_end, x_start = np.where(mask == 1)[0].max(), np.where(mask == 1)[0].min()
    y_end, y_start = np.where(mask == 1)[1].max(), np.where(mask == 1)[1].min()

    return y_start, x_start, y_end, x_end


def get_extremities_depth_delta(dmap, mask, width_ratio=0.1):
    """
    Computes the difference of depth between 

    Input:
        - dmap : np.array (H, L) depth map
        - mask : np.array (L, H) mask
        - width ratio: portion of the mask bounding box to 
        crop the tail & head

    Output:
        - delta : float, difference of depth between the depth 
        map croppend on tail & head
    """
    y_start, x_start, y_end, x_end = get_bb_from_mask(mask)
    width_bb = int(abs(y_end - y_start) * width_ratio)
    mean_ex1 = dmap.T[np.where(mask[x_start:x_end,
                                    y_start:(y_start + width_bb)] == 1)].mean()
    mean_ex2 = dmap.T[np.where(mask[x_start:x_end,
                                    (y_end - width_bb):y_end])].mean()

    return abs(mean_ex1 - mean_ex2)

def barycentric_length_estimation(mask, dmap, width_ratio=0.05):
    """
    Computes the barycenter of head & tail of the fish converts
    it into real world points then calculate the euclidean norm

    Input:
        - mask: np.array of size (W, H)
        - dmap: np.array of size (W, H)
        - width_ratio: ratio of the bounding box to calculate the mean
        depth of tail & head extremities

    Output:
        - pred_length: float
    """
    y_start, x_start, y_end, x_end = get_bb_from_mask(mask)
    width_bb = int(abs(y_end - y_start) * width_ratio)
    # Isolates head & tail
    extremity1 = list(np.where(mask[:, y_start:(y_start + width_bb)] == 1))
    extremity2 = list(np.where(mask[:, (y_end - width_bb):y_end] == 1))
    extremity1[1] += y_start
    extremity2[1] += (y_end - width_bb)
    # Calculates barycenter
    # The dmap needs to be converted in cm
    y1, x1 = extremity1
    y2, x2 = extremity2
    wx1, wy1, wz1 = convert_to_world_point(x1, y1, dmap *10)
    wx2, wy2, wz2 = convert_to_world_point(x2, y2, dmap * 10)
    b1 = wx1.mean(), wy1.mean(), wz1.mean()
    b2 = wx2.mean(), wy2.mean(), wz2.mean()
    # Computes the norm
    pred_length = np.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])** 2 + (b1[2] - b2[2])**2).mean()

    return pred_length

def pca_length_estimation(mask, dmap, width_ratio=0.05):
    """
    Performs PCA projection in 1D of the 3D points cloud of the fish 
    and computes the norm of further projected fish points distribution
    The number of projected fish points is defined by w_ratio
    
    Input:
        - mask: np.array of size (W, H)
        - dmap: np.array of size (W, H)
    
    Output:
        - pred_length: float
    """
    y, x = np.nonzero(mask)
    wx, wy, wz = convert_to_world_point(x, y, dmap * 10)
    fish_cloud = np.array([wx, wy, wz]).T
    fish_cloud -= np.mean(fish_cloud, axis=0)
    pca = decomposition.PCA(n_components=1)
    pca.fit(fish_cloud)
    X = pca.transform(fish_cloud)
    X = np.sort(X, axis=0)
    max_points = int(len(X) * width_ratio)
    X = np.sort(X, axis=0)
    X_start = X[:max_points]
    X_end = X[-max_points:]
    norm = []
    for i in range(len(X_start)):
        norm.append(np.sqrt((X_start[i]-X_end[-(i+1)])**2))
    pred_length = sum(norm) / len(norm)
    
    return pred_length