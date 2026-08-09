# System Calibration

PolyUMI depends on some hardcoded constants to run inference. Where possible, scripts are included in the repo to produce correctly calibrated values for these.
Some of these constants
should generally be the same from setup to setup (ie the gripper width offset), 
while others may be substantially different (sensor + arm execution latencies). 

Here is a quick doc detailing the calibration procedures for the system & where to find the associated values:


