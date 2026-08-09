# System Calibration

PolyUMI depends on some hardcoded constants to run inference. Where possible, scripts are included in the repo to produce correctly calibrated values for these.
Some of these constants
should generally be the same from setup to setup (ie the gripper width offset), 
while others may be substantially different (sensor + arm execution latencies). 

Here is a quick doc detailing the calibration procedures for the system & where to find the associated values:

## Geometric Constants
### Hand -> TCP frame

The TCP or "Tool Control Point" frame is the point whose pose we control with the model output. For PolyUMI (as in the original UMI) it is the point positioned in between the two gripper fingertips, level with the fingers' upper surface.

The offset between this point and the built-in franka hand frame (or the hand frame of whatever arm you're using) is both arm & end-effector specific, and must be recalibrated for every update to PolyUMI's design + for every new platform it supports.

**TODO describe how to calibrate & set**

### Gripper Static Offsets

The PolyUMI gripper & the arm-mounted PolyUMI end-effector will read off finger positions differently, and also may have different ranges of motion in practice.

We assume that the preprocessing tool *pingest* (which measures the finger width using ArUco tags) and the arm (which likely measures with an encoder of some sort) both report metrically correct & consistent values with some static offset (i.e. a mm is a mm, less some constant).

**TODO describe how to calibrate & set**

## Latencies

- [ ] sensor latencies:
    - [ ] gopro -> ros inference client node
    - [ ] finger cam -> ros inference client node
    - [ ] piezo mic -> ros inference client node
    - [ ] EE cartesian pose -> ros inference client node
- [ ] ros -> inference server -> response (round trip delay of server call -- currently missing in inference.yaml)
- [ ] arm execution delay