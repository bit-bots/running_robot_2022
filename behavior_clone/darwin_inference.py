import time

import cv2
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

import os
import cv2
import numpy as np
import datetime

from behavior_clone.model import EyePredModel1, MLPSeq

import controller
from controller import Robot
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


CAMERA_DIVIDER = 50
ACTION_CHARACTERS = ['w', 'a', 's', 'd', 'n']
CHECKPOINT = 'C:\\Users\\florian\\rrc\\running_robot_2022\\checkpoints\\model_epoch_200_step_129.pth' 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Load model")
img_size = (120, 160)
max_len = 5
model = MLPSeq(img_size=img_size, token_size=128, max_len=max_len)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Show summary
print("Model during inference:")
print(model)

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

def run_motion(file, robot, joints):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        curr_time = 0.0
        for row in reader:
            duration_str = row["#WEBOTS_MOTION"].split(":")
            # Duration in sec
            duration = int(duration_str[0]) * 60 + int(duration_str[1]) + int(duration_str[2]) / 1000
            duration -= curr_time
            curr_time += duration
            diffs = {}
            for joint in joints:
                current_postion = robot.getDevice(joint  + 'S').getValue()
                diffs[joint] = (current_postion, float(row[joint]) - current_postion)
            # Wait for duration
            for i in range(int(round((duration * 1000) / timestep))):
                for joint in joints:
                    interpolated_angle = diffs[joint][0] + diffs[joint][1] * ((i + 1) / int(round((duration * 1000) / timestep)))
                    motor = robot.getDevice(joint)
                    motor.setVelocity(motor.getMaxVelocity())
                    motor.setPosition(float(interpolated_angle))
                robot.step(timestep)

motionManager.playPage(1)
run_motion("C:\\Users\\florian\\rrc\\running_robot_2022\\animations\\TimonsRolle.motion", robot, positionSensorNames)
motionManager.playPage(11)

counter = 0
while True:
    for i in range(CAMERA_DIVIDER):
        robot.step(timestep)
        gaitManager.step(basicTimeStep)

    image = camera.getImage()
    np_image = np.frombuffer(image, np.uint8).reshape(
        camera.getHeight(), 
        camera.getWidth(),
        4)[..., :3]

    data = torch.from_numpy(np_image.transpose(2, 0, 1)).float().unsqueeze(0) / 255

    data = data.to(DEVICE)

    output = torch.argmax(model(data)[0])
    print(output)

    action = ACTION_CHARACTERS[int(output.detach().long().cpu())]
    print(action)

    if action == 'r':
        gaitManager.setXAmplitude(0.5)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
        for i in range(500):
            robot.step(timestep)
            gaitManager.step(basicTimeStep)
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
        motionManager.playPage(9)
        for i in range(30):
            robot.step(timestep)
        motionManager.playPage(1)
        run_motion(os.path.join("animations", "TimonsRolle.motion"), robot, positionSensorNames)
        for i in range(30):
            robot.step(timestep)
        motionManager.playPage(1)
        for i in range(30):
            robot.step(timestep)
        motionManager.playPage(11)
    elif action == 'w':
        gaitManager.setXAmplitude(1.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
    elif action == 'a':
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.5)
    elif action == 's':
        gaitManager.setXAmplitude(-0.5)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
    elif action == 'd':
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(-0.5)
    elif action == 'n':
        gaitManager.setXAmplitude(0.0)
        gaitManager.setYAmplitude(0.0)
        gaitManager.setAAmplitude(0.0)
    else:
        print("unknown action")