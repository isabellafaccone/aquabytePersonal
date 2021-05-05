# Multiple-Fish Tracking Playground (MFT-PG)

This project aims to study:
 * Fish detection on the TX2
 * Simple multiple-fish tracking 
 * Selection of optimal tracklets for biomass prediction
 




sketchpad

datasets
docker
training
detection
tracking
tube-scorer

this convert to onnx https://github.com/jkjung-avt/tensorrt_demos 

tracker tests
 ** test detector drop rate!  and how tracker is robust to it!!
 ** test FPS ablation rate with that too


want:
 * yolo models for all sizes multiple-of-16; just run small max batch for now
 * compute average recall for all of them
 * compute mean inference for all of them
 * train in parallel (eventually)

 * a prediction script that does tensorRT

 * note: can do scripts as separate runs or all in one run!!
 



### Tickets

  * https://aquabyte.atlassian.net/browse/ENGALL-2424
  * https://aquabyte.atlassian.net/browse/ENGALL-2425
  * https://aquabyte.atlassian.net/browse/ENGALL-2426
