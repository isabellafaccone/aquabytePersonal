'''
    File name         : sealice_candidate_detectors_1.py
    File Description  : Detect candidate sealice for classification
    Author            : Sudarshan Ramenahalli
    Date created      : 01/10/2018
    Date last modified: 01/11/2018
    Python Version    : 2.7.12
'''

# Import python libraries
import numpy as np
import cv2
import skimage
from skimage.measure import label, regionprops

# set to 1 for pipeline images
debug = 0

class SealiceCandidates(object):
    """Sealice candidate selection
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by SealiceCandidates class
        Args:
            None
        Return:
            None
        """
        # Sea lice are extremely small and bright compared to iamge size and 
        #the size of a fish. 
        # So, any region with intensity > self.mask_thresh, reasonably small
        # (area of the region between 25 and 150 pixels) and elongated 
        #(eccentricity between o.75 and 1) is 
        # considered as a sea lice candidate 
        self.area_min = 50
        self.area_max = 150
        self.ecc_min = 0.85
        self.ecc_max = 1.0
        self.mask_thresh = 100
        self.bbox_halfwidth = 20

    def Detect(self, frame):
        """Detect sealice candidate locations in video frame using the following pipeline
            - Convert captured frame from BGR to GRAY
            - Intensity based threshold to get binary mask
            - Label binary mask for connected components
            - Retain only those connected components which have: (1) 25 < area < 150 pixels and (2) 0.75 < eccentricity < 1.0 
            - Find centroids for each valid connected component
        Args:
            frame: single video frame, RGB 
        Return:
            candidate_locations: List of object centroids in a frame
        """

        # Convert BGR to GRAY if not already converted.
        if len(np.shape(frame)) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        nr, nc = np.shape(gray)
        ret,thresh_mask = cv2.threshold(gray,self.mask_thresh,255,cv2.THRESH_BINARY)
        thresh_mask=np.where(thresh_mask==255,1,0)
        
        labels = skimage.measure.label(thresh_mask,background=0)
        regions = skimage.measure.regionprops(labels)
        
        candidate_locs = []
        for region in regions:
            if region.area > self.area_min and region.area < self.area_max:
                if region.eccentricity > self.ecc_min and region.eccentricity < self.ecc_max:
                    if region.centroid[0]-self.bbox_halfwidth > 0 \
                        and region.centroid[0]-self.bbox_halfwidth < nr \
                        and region.centroid[1]-self.bbox_halfwidth > 1 \
                        and region.centroid[1]-self.bbox_halfwidth < nc:
                            b = np.array([[region.centroid[1]], [region.centroid[0]]])
                            candidate_locs.append(np.round(b))
                            
        return candidate_locs