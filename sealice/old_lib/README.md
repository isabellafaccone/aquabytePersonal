### Folder structure here:

### Archive folders:

- annotations - Files relating to parsing annotations file, as well as writing sample bounding boxes. None of these seem to be critical files, unless we use iSeg again to process the annotations.

- optical_flow - Files relating to optical flow and experiments. Non-critical until we starting using optical flow for segmentation across frames.

- tracking - Files relating to kalman filtering and tracking of fish across frames. Non-critical until we starting using tracking across frames.

- scratch_files - Some scratch files with non-production code.

- segmentation - Files such as Mask R-CNN that was used for segmentation. Seems to be duplicate of existing code elsewhere.

### Critical folders:

- models - Contains trained models, for example yml files for the SVM model

- future_versions - Contains code that has some ideas that be incorporated into the current version of the code

### Main code:

#### 1. features_from_video.py

This file should be used to gather feature descriptors from input training videos with annotations. ORB features are implemented right now. 

For better performance, HAAR and Local Binary Patterns (LBP) should also be explored, if needed. The file extracts image patches (RGB and gray), rotated image patches based on the annotation (RGB and gray) and ORB feature descriptors around locations annotated as ‘lice’ or ‘non_lice’. 

The extracted features are saved as .npy files used for analysis or training an SVM classifier.

#### 2. sealice_svm_train.py 

This file assumes required data, ie, ORB feature descriptors for  ‘lice’ and ‘non_lice’ classes are available as .npy files. The file reads the .noy files, organizes the data in the required format and trains an SVM classifier. 

Since the number of ‘non_lice’ examples are very few (~3000 non_lice examples compared to ~10,000 examples for lice class), only a portion of lice examples are randomly selected from the 10,000 lice examples for training. This is to make sure, both classes are in equal proportion. Multi-resolution grid search with 10-fold cross-validation is used for training the ORB feature descriptor based SVM classifier. Radial Basis Function kernels are used. The two hyper-parameters $C$ and $\gamma$ are tuned using the multi-res approach. 

The default grid can be obtained and values changed using the following code. Only example for $C$ is below:
```
C_grid = mdl.getDefaultGridPtr(cv2.ml.SVM_C)
        C_grid.minVal=1e-10
        C_grid.maxVal=1e10
        C_grid.logStep=2
C_grid.logStep is the step interval in which grid values are incremented. The grid steps are computed as follows:
[C_grid.minVal, C_grid.minVal*C_grid.logStep, C_grid.minVal*C_grid.logStep^2, C_grid.minVal*C_grid.logStep^3, …, C_grid.maxVal]
```

So, the grid resolution is controlled by C_grid.logStep. The grid can be refined using this parameter based on desired accuracy and the accuracy obtained using the SVM classifier. 

svm.trainAuto function of OpenCV does the training where 10-fold cross-validation is carried out by the svm.trainAuto function, so no need to do additional cross-validation. 

Depending on grid resolution, number of training examples, range of values (minVal and maxVal), training can take a lot of time. For the above values, it takes about ~12 hours on Intel Core i7 machine with 8GB RAM (my laptop). 

Once training is completed, the trained SVM model is saved in a .yml file in the same folder. 

#### 3. sealice_svm_predict_anno_video.py

This is the validation step to verify if the trained SVM classifier is working as expected. Since OpenCV version of SVM classifier does not allow us to access the cross-validation score during training, this step becomes essential. 

In this step, only the already annotated locations are classified using the trained model to compute the precision and recall metrics. 

In this step, the results are shown on each frame of video for better visualization. Each location, will have two squares. The inner square represents the ground truth, based on the annotation. If the square is green, it denotes a lice location, red indicates a non-lice location. The outer square represents the classification result, with same color coding. So, if a location is correctly classified, you will see two squares of matching color at that location, irrespective of whether it is red (for non-lice) or green (for lice). 

#### 4. detect_sealice_svm_orb.py

This is the last step, where given a video, sea lice key points or possible sea lice candidate locations are automatically extracted and the classification of each candidate location is done. 

If a candidate location is detected as sea lice, it will be visualized in the video with a blue square around that location. Sea lice key points are currently chosen based on a simple threshold. On the gray image, anything with a gray value above 100 is set to +1 in a binary mask. The binary mask in labeled for connected components, then each connected component which has  eccentricity above 0.75 and area between 150 - 250 pixels is chosen as a candidate sealice location. These parameters need to be tuned more carefully. Also, a different method based on the properties of sealice patches should be used in this step. 
