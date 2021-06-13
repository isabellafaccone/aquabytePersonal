# Multiple-Fish Tracking Playground (MFT-PG)

This project aims to study:
 * Fish detection on the TX2
 * Simple multiple-fish tracking 
 * Selection of optimal tracklets for biomass prediction
 
## Quickstart

Use the `mft-cli` program to set up the playground and dockerized
runtime environments. Try `$ ./mft-cli --help` to start.


## JIRA Tickets

### Algorithmic & Computational Feasibility: Detector
  * Prove that YOLO+SORT can run on the TX2 via TensorRT https://aquabyte.atlassian.net/browse/ENGALL-2424
  * Train a TX2-friendly fish detector using data from next-gen optics https://aquabyte.atlassian.net/browse/ENGALL-2426
  * Study the accuracy vs latency pareto for YOLO detection on the TX2 https://aquabyte.atlassian.net/browse/ENGALL-2651

### Algorithmic & Computational Feasibility: Tracking & Tracklet Scoring
  * Study the robustness of a tracker (e.g. SORT) at lower frame rates (e.g. 5-8FPS target) https://aquabyte.atlassian.net/browse/ENGALL-2425
  * Devise and test a tracklet scorer https://aquabyte.atlassian.net/browse/ENGALL-2650
  * Rough "back-test" of tracklet scorer: compare tracklet scores vs AKPD scores on past data https://aquabyte.atlassian.net/browse/ENGALL-2711
  * Rough "forward-test" of tracklet scorer: compare tracklet scores vs AKPD scores on (unlabeled) GoPro data https://aquabyte.atlassian.net/browse/ENGALL-2712

### Production-oriented Demo
  * Production-like docker image for Detector+Tracker https://aquabyte.atlassian.net/browse/ENGALL-2677
  * Production-like host image for Detector+Tracker https://aquabyte.atlassian.net/browse/ENGALL-2676
  * Demo deployment of Detector + Tracker (or just Detector + Tracklet scorer) in production-like compute environment https://aquabyte.atlassian.net/browse/ENGALL-2647
  * Perform a parallel test (or shadow-launch) of Detector + Tracklet Scorer as an "upload rejector" https://aquabyte.atlassian.net/browse/ENGALL-2713 



#### pwais scratchpad
 * tracks df has track ids added.
 * dropping a track or end of track triggers a tracklet score run and tracklet gets score
 * add in eval a tracklet miner
 * fish counts per hour!!!
 * MOTS is TBD ...