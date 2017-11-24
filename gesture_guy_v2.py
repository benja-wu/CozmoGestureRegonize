#!/usr/bin/env python3
# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.models import model_from_json 
from keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
from cozmo.util import degrees, distance_mm, speed_mmps
import shutil  
import json
import sys
import cv2 
import asyncio
from cozmo.util import degrees

sys.path.append('../lib/')
import cozmo
import time 


labels_want = ['Swiping Left',
          'Swiping Right',
          'Swiping Down',
          'Swiping Up',
          'Nothing']

image_size=(88,50)

class GestureBrain():

    def __init__(self,  model ):
        self.curr_stream = []
        self.model = model 
        self.counter  = 0 

    def adjust_judge_sequence(self):
        frame_diff = len(self.curr_stream) - 40 
        print("diff is", frame_diff)
        if  frame_diff == 0 :
            return self.curr_stream
        elif frame_diff > 0 :
            return self.curr_stream[frame_diff:]
        else :
            return self.curr_stream[:1] * abs(frame_diff) + self.curr_stream

    def prepreocess_img (self,  img_array):
        return (img_array / 255. )

    def regonize(self):

        x = []         
        x.append(self.curr_stream)

        X = np.array(x)
        #result = self.model.predict(X)
        index = self.model.predict_classes(X)
        #print(result)
        #self.curr_stream = []
        if index != 4 :
           print("cleaning!")
           self.curr_stream = []

        return index 

    def img_num (self ):
        return len( self.curr_stream)

    # build a first in first out queue for img
    # and keep 40 frames length 
    def push_img (self,  img ):
        img = self.prepreocess_img(img )
        self.curr_stream.append(img)

        if len( self.curr_stream ) > 40 :
            self.curr_stream.pop(0)

def adjust_sequence_length( frame_files):
    """Adjusts a list of files pointing to video frames to shorten/lengthen
    them to the wanted sequence length (self.seq_length)"""
    frame_diff = len(frame_files) - 40 
    if frame_diff == 0:
        # No adjusting needed
        return frame_files
    elif frame_diff > 0:
        # Cuts off first few frames to shorten the video
        return frame_files[frame_diff:]
    else:
        # Repeats the first frame to lengthen video
        return frame_files[:1] * abs(frame_diff) + frame_files

def preprocess_image(image_array):
        return (image_array / 255. )

def build_sequence( ):
    path = "./TestImgv3/"
    frame_files = os.listdir(path)
    # add sorted, so we can recognize the currect sequence
    frame_files = sorted(frame_files)
    print(frame_files)
    sequence = []

    # Adjust length of sequence to match 'self.seq_length'
    frame_files = adjust_sequence_length(frame_files)

    frame_paths = [os.path.join(path, f) for f in frame_files]
    for frame_path in frame_paths:
        image = load_img(frame_path, target_size=image_size)
        image_array = img_to_array(image)
        image_array = preprocess_image(image_array)

        sequence.append(image_array)

    return np.array(sequence)

def check_value_pics( model):
    x = [] 
    sequence = build_sequence()
    x.append(sequence)

    X = np.array(x)
    #print("---- val --- ",model.predict(X))
    index = model.predict_classes(X)

    print("----- we guess is -----",labels_want[index[0]])

    return index

# version 2 
async def gesture_guy(robot):
    file_object = open('/Users/benja/code/jester/models/smallc3d_architecture_v3.json')
    model_json = file_object.read( )
    model = model_from_json(model_json)  
    model.load_weights('/Users/benja/code/jester/models/smallc3d_weights_v3.h5')    

    gb = GestureBrain(model)
    print("start ======")

    counter = 0 
    while True :
        counter += 1

        await asyncio.sleep(0.02)
        screen = np.array(robot.world.latest_image.raw_image.resize(image_size))

        gb.push_img(screen)

        if counter != 40 :
            continue 
        else:
            counter = 0 

        action = gb.regonize()
       
        print("predict type :", action )

        if action == 0 :
            await robot.say_text("move left!").wait_for_completed()
            await robot.turn_in_place(degrees(-20), in_parallel = True).wait_for_completed()

        elif action == 1:
            await robot.say_text("move right!").wait_for_completed()
            await robot.turn_in_place(degrees(20), in_parallel = True ).wait_for_completed()

        elif action == 2: # move down the lift 
            #await  self.cozmo.move_lift(-1*4, in_parallel = True ).wait_for_completed()
            await robot.say_text("move down!").wait_for_completed()
            #robot.move_head(-1*4)
            await robot.drive_straight(distance_mm(-50), speed_mmps(50)).wait_for_completed()
            await robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
        elif action == 3: # move up the lift 
            #await robot.turn_in_place(degree(360), in_parallel = True ).wait_for_completed()
            await robot.say_text("move up!").wait_for_completed()

            await robot.drive_straight(distance_mm(50), speed_mmps(50)).wait_for_completed()
            await robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
        else :
            await robot.say_text("nothing!").wait_for_completed()

async def seeing_guy(robot):
    file_object = open('/Users/benja/code/jester/models/smallc3d_architecture_v3.json')
    model_json = file_object.read( )
    model = model_from_json(model_json)  
    model.load_weights('/Users/benja/code/jester/models/smallc3d_weights_v3.h5')    

    index = 0 



    #os.mkdir('./TestImgv3')    
    while True:
        screen  = np.array(robot.world.latest_image.raw_image)

        name = str(index).zfill(5)
        file_name =  name +".jpg"

        cv2.imwrite('./TestImgv3/'+file_name, screen)

        index += 1
        print("get pic ",index)

        await asyncio.sleep(0.08)

        if index == 41 :
            action =  check_value_pics( model)

            if action == 0 :
                   action2 = robot.turn_in_place(degrees(-30), in_parallel=True)
                   action2.wait_for_completed()
            elif action == 1 :
                   action2 = robot.turn_in_place(degrees(30), in_parallel=True)
                   action2.wait_for_completed()
            elif action == 2 :
                   action2 = robot.turn_in_place(degrees(180), in_parallel=True)
                   action2.wait_for_completed()
            else :
                   action2 = robot.turn_in_place(degrees(360), in_parallel=True)
                   action2.wait_for_completed()
            
            
            index = 0 
            input()
            shutil.rmtree('./TestImgv3')
            os.mkdir('./TestImgv3')
        

async def run(sdk_conn):

    robot = await sdk_conn.wait_for_robot()

    # Turn on image receiving by the camera

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled  = True
    await robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
    
    try:
        await gesture_guy(robot)

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
         
        

if __name__ == '__main__':
    cozmo.setup_basic_logging()
    cozmo.robot.Robot.drive_off_charger_on_connect = False  # RC can drive off charger if required
    try:
        cozmo.connect(run)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
