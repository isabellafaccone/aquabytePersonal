
import numpy as np

DORSAL_BACK = "DORSAL_BACK"
VENTRAL_BACK = "VENTRAL_BACK"
DORSAL_MID = "DORSAL_MID"
VENTRAL_MID = "VENTRAL_MID"
DORSAL_FRONT = "DORSAL_FRONT"
VENTRAL_FRONT = "VENTRAL_FRONT"
HEAD = "HEAD"


def get_kp_location(kps, keypointType):
    """
    Parameters: 
    ----------
    keypointType : str
        one of the 8 key points
        
    Returns:
    ----------
    np.array: x, y from kps
    """
    for kp in kps:
        if kp["keypointType"] == keypointType:
            return np.array((kp['xCrop'], kp['yCrop']))

def face_left(kps):
    "Check if fish is facing left"
    ul = get_kp_location(kps, "UPPER_LIP")
    tn = get_kp_location(kps, "TAIL_NOTCH")
    return ul[0] < tn[0]

def is_between_v(p, a, b):
    """
    Check whether vector p is between vector a and b
    i.e. check if cross product/sin(theta) has the same singe:
        a X b * a X p >= 0 and b X a * b X p >= 0
    Parameters: 
    ----------
    p, a, b: np array (2,) 
    
    """

    return (((a[1] * b[0] - a[0] * b[1]) * (a[1] * p[0] - a[0] * p[1]) >= 0) 
           and ((b[1] * a[0] - b[0] * a[1]) * (b[1] * p[0] - b[0] * p[1]) >= 0))


def is_between_p(P, A, B, O):
    """
    Check whether point P is between point A and B with O as origin
    Parameters: 
    ----------
    P, A, B, O: np array (2,)   
    """
    a = A - O
    b = B - O
    p = P - O

    return is_between_v(p, a, b)

def get_auxiliary_kps(kps):
    """return auxiliary key points for fish sector
    h0, h1: upper and lower point of head
    """
    ad_fin = get_kp_location(kps, "ADIPOSE_FIN")
    an_fin = get_kp_location(kps, "ANAL_FIN")
    ad_an_mid = np.average(np.array([ad_fin, an_fin]), axis = 0)

    ds_fin = get_kp_location(kps, "DORSAL_FIN")
    pv_fin = get_kp_location(kps, "PELVIC_FIN")
    # ds_pv_mid = np.average(np.array([ds_fin, pv_fin]), axis = 0)

    pt_fin = get_kp_location(kps, "PECTORAL_FIN")


    h1 = 1 / 8 * (pv_fin - pt_fin) + pt_fin
    h0 = h1 + 0.7 * (ds_fin - pv_fin)
    
    h_mid = np.average(np.array([h0, h1]), axis = 0)
    
    pv_back = 1 / 3 * (an_fin - pv_fin) + pv_fin
    ds_back = pv_back + 1 /2 * (ds_fin - pv_fin + ad_fin - an_fin)
    ds_pv_mid = np.average(np.array([ds_back, pv_back]), axis = 0)
    return {"ad_an_mid": ad_an_mid, 
            "ds_pv_mid": ds_pv_mid, 
            "h0": h0, 
            "h1": h1, 
            "h_mid": h_mid,
            "pv_back": pv_back,
            "ds_back": ds_back
           }

def get_sector(p, kps):
    """
    Parameters: 
    ----------
    p : np.array of point coordinate
    get the sector point p falls into
    
    Returns:
    ----------
    str: which sector point p is at
    """
    eye = get_kp_location(kps, "EYE")
    tn = get_kp_location(kps, "TAIL_NOTCH")
    ad_fin = get_kp_location(kps, "ADIPOSE_FIN")
    an_fin = get_kp_location(kps, "ANAL_FIN")
    ds_fin = get_kp_location(kps, "DORSAL_FIN")
    pv_fin = get_kp_location(kps, "PELVIC_FIN")
    pt_fin = get_kp_location(kps, "PECTORAL_FIN")

    aux_kps = get_auxiliary_kps(kps)
    
    ad_an_mid = aux_kps["ad_an_mid"]
    ds_pv_mid = aux_kps["ds_pv_mid"]
    h1 = aux_kps["h1"]
    h0 = aux_kps["h0"]
    h_mid = aux_kps["h_mid"]

    if is_between_p(p, ad_fin, tn, ad_an_mid): return DORSAL_BACK 
    elif is_between_p(p, an_fin, tn, ad_an_mid): return VENTRAL_BACK
    elif is_between_p(p, ds_fin, ad_an_mid, ds_pv_mid): return DORSAL_MID
    elif is_between_p(p, pv_fin, ad_an_mid, ds_pv_mid): return VENTRAL_MID
    elif is_between_p(p, h0, ds_pv_mid, h_mid): return DORSAL_FRONT
    elif is_between_p(p, h1, ds_pv_mid, h_mid): return VENTRAL_FRONT
    else: return HEAD
    
    
    