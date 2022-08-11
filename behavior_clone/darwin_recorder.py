import os
import cv2
import numpy as np
import datetime

import controller
from controller import Robot
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


CAMERA_DIVIDER = 50
OUT_FOLDER = 'gen_data' 

now = datetime.datetime.now()
run_name = f"run_{now.strftime('%Y_%m_%d__%H_%M_%S')}"

working_dir = os.path.join(OUT_FOLDER, run_name)
if not os.path.exists('my_folder'):
    os.makedirs(working_dir)

# Create the Robot instance.
robot = Robot()
camera = robot.getDevice('Camera')
gyro = robot.getDevice('Gyro')


basicTimeStep = int(robot.getBasicTimeStep())
# Initialize motion manager.
motionManager = RobotisOp2MotionManager(robot)
# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

camera.enable(timestep * CAMERA_DIVIDER)
gyro.enable(basicTimeStep)

positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                       'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                       'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                       'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                       'FootR', 'FootL', 'Neck', 'Head')
for sensor in positionSensorNames:
    robot.getDevice(sensor + 'S').enable(basicTimeStep)

# Perform one simulation step to get sensors working properly.
robot.step(timestep)

# Initialize OP2 gait manager.
gaitManager = RobotisOp2GaitManager(robot, "")
gaitManager.start()
gaitManager.setXAmplitude(0.0)
gaitManager.setYAmplitude(0.0)
gaitManager.setBalanceEnable(True)

counter = 0
while True:
    for i in range(CAMERA_DIVIDER):
        robot.step(timestep)
        gaitManager.step(basicTimeStep)
    
    image = camera.getImage()
    np_image = np.frombuffer(image, np.uint8).reshape(
        camera.getHeight(), 
        camera.getWidth(),
        4)

    action = "n"

    cv2.imshow("Image", np_image)
    key = cv2.waitKey(0)
    
    if key == ord('q'):
        break
    if key == ord('w'):
        gaitManager.setXAmplitude(1.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
        action = "w"
    elif key == ord('a'):
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.5)
        action = "a"
    elif key == ord('s'):
        gaitManager.setXAmplitude(-0.5)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
        action = "s"
    elif key == ord('d'):
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(-0.5)
        action = "d"
    else:
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
    # Save data
    cv2.imwrite(os.path.join(working_dir, f'frame_{counter:06d}_{action}.png'), np_image)
    counter += 1

cv2.destroyAllWindows()
