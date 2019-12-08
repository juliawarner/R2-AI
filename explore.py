#R2-D2 random exploration program
#R2 will randomly explore environment while preforming object detection. 
#If R2 sees Leia, he will make a greeting sound. 
#If R2 sees Obi-Wan, he will make a greeting sound or deliver Leia's message if he has seen Leia already.
#If R2 sees Darth Vader, he will scream and turn around and move in the opposite direction for a while. 
#Object detection code is from Evan Juras' tensorflow on Raspberry Pi tutorial. 
#https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi
#See also code author's notes below. 


######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

from gpiozero import Robot
from gpiozero import RGBLED
import pygame
import random

#R2-D2 class
class R2_D2():
    def __init__(self):
        #movement variables
        self.legs = Robot(left=(5, 6), right=(17, 27))
        self.turning_right = False
        self.turning_left = False
        self.moving_forward = False
        self.moving_backward = False

        #used when preforming specific actions for fleeing Darth Vader
        self.turning_around = False
        self.time_turning_around = 0
        self.running_away = False
        self.time_running_away = 0
        self.fleeing = False

        #used to blink red-blue light
        self.light = RGBLED(red=16, green=20, blue=21)
        self.light.color = (1, 0, 0)
        self.red = True
        self.time_since_last_blink = 0

        #used to make random movements
        self.time_stopped = 0
        self.time_moving = 0
        self.moving = False

        #used for reaction to characters
        self.seeing_Leia = False
        self.seen_Leia = False
        self.time_seeing_Leia = 0
        self.seeing_Obiwan = False
        self.time_seeing_Obiwan = 0
        self.seeing_Vader = False
        self.time_seeing_Vader = 0

    #function calls Robot.right() if not already moving right
    def turn_right(self):
        if(not self.turning_right):
            self.legs.right()
            self.turning_right = True

    #funtion calls Robot.left() if not already moving left
    def turn_left(self):
        if(not self.turning_left):
            self.legs.left()
            self.turning_left = True

    #function calls Robot.forward() if not already moving forward
    def move_forward(self):
        if(not self.moving_forward):
            self.legs.forward()
            self.moving_forward = True

    #function calls Robot.backward() if not already moving backward
    def move_backward(self):
        if(not self.moving_backward):
            self.legs.backward()
            self.moving_backward = True

    #functions stops all robot movements, sets all movement variables to false
    def stop_movement(self):
        self.turning_right = False
        self.turning_left = False
        self.moving_forward = False
        self.moving_backward = False
        self.legs.stop()

    #plays a sound based on given input, does nothing if input invalid
    #assumes pygame is initialized 
    def play_sound(self, sound_name):
        sound_location = './sounds/' + sound_name + '.wav'
        sound = pygame.mixer.Sound(sound_location)
        pygame.mixer.Sound.play(sound)

    #updates all time dependent component of R2-D2
    #expected delta_time in seconds
    def update(self, delta_time):
        #update light
        self.update_light(delta_time)

        #update character sightings
        self.update_character_sighting(delta_time)

        #check if now completing special fleeing movements
        if(self.fleeing):
            self.update_fleeing(delta_time)
        else:
            #we are not fleeing, continue making random movements
            self.update_random_movement(delta_time)

    #updates the times of seeing different characters (so R2 doesn't react more than once when seeing someone)
    #expected delta_time is in seconds
    def update_character_sighting(self, delta_time):
    	if(self.seeing_Leia):
    		self.seen_Leia = True
    		#if this is the first time we're seeing her (for now), say hello!
    		if(self.time_seeing_Leia == 0):
    			self.play_sound('cute')
    		self.time_seeing_Leia = self.time_seeing_Leia + delta_time
    	else:
    		self.time_seeing_Leia = 0

    	if(self.seeing_Obiwan):
    		#if this is the first time we're seeing him (for now), say hello of deliver Leia's message
    		if(self.time_seeing_Obiwan == 0):
    			if(self.seen_Leia):
    				self.play_sound('helpme_short')
    			else:
    				self.play_sound('excited')
    		self.time_seeing_Obiwan = self.time_seeing_Obiwan + delta_time
    	else:
    		time_seeing_Obiwan = 0

    	if(self.seeing_Vader):
    		#if this is the first time we're seeing him (for now), run away!
    		if(self.time_seeing_Vader == 0):
    			self.fleeing = True
    		self.time_seeing_Vader = self.time_seeing_Vader + delta_time
    	else:
    		time_seeing_Vader = 0
    
    #decides whether to stop or choose a new random movement
    #stops last for 3 seconds, movements last for 3 seconds
    #expected delta_time in seconds
    def update_random_movement(self, delta_time):
        if(self.moving):
            #update time
            self.time_moving = self.time_moving + delta_time

            #check to see if we should stop moving
            if(self.time_moving >= 3.0):
                #reset everything and stop moving
                self.time_moving = 0
                self.moving = False
                self.stop_movement()
        else:
            self.time_stopped = self.time_stopped + delta_time

            #check to see if it's time for a new random movement
            if(self.time_stopped >= 3.0):
                #reset stopping variables
                self.time_stopped = 0
                self.moving = True

                #choose a new random movement
                random_movement = random.randint(1, 5)
                if(random_movement == 1):
                    self.move_backward()
                elif(random_movement == 2):
                    self.turn_right()
                elif(random_movement == 3):
                    self.turn_left()
                else:
                    self.move_forward()

    #updates R2's fleeing movements
    #expects delta_time in seconds
    def update_fleeing(self, delta_time):
        if(self.turning_around):
            if(self.time_turning_around == 0):
                #we just saw Darth Vader!
                self.stop_movement()
                self.play_sound('scream')
                self.turn_right()
            else:
                #check if we should stop turning 
                if(self.time_turning_around >= 2.0):
                    #stop turning, start running
                    self.turning_around = False
                    self.time_turning_around = 0

                    self.running_away = True

            #update time if we are still turning around
            if(self.turning_around):
                self.time_turning_around = self.time_turning_around + delta_time

        if(self.running_away):
            if(self.time_running_away == 0):
                #we just started running away
                self.move_forward()
            else:
                #check if we should stop running
                if(self.time_running_away >= 3.0):
                    #stop running
                    self.running_away = False
                    self.fleeing = False
                    self.time_running_away = 0

            #update time if we are still running away
            if(self.running_away):
                self.time_running_away = self.time_running_away + delta_time


    #keeps track of time since last light blink and blinks light
    #if that time is greater than 1 second. 
    #delta time parameter is in seconds
    def update_light(self, delta_time):
        self.time_since_last_blink = self.time_since_last_blink + delta_time

        #blink about every 1 second
        if(self.time_since_last_blink >= 1.0):
            self.time_since_last_blink = 0
            if(self.red):
                self.red = False
                self.light.color = (0, 0, 1)
            else:
                self.red = True
                self.light.color = (1, 0, 0)

#constant fo match threshold for R2 to react to a detection
THRESHOLD = 0.85

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    # Initialize pygame, used for playing sounds
    pygame.init()

    # Initialize R2D2
    artoo = R2_D2()

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        print('Classes:')
        print(classes)
        print('Scores:')
        print(scores)

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        #check to see if R2 sees anyone 
        # Check the class of the top detected object by looking at classes[0][0].

        #check if seeing Vader
        if ((int(classes[0][0]) == 1) and (scores[0][0] > THRESHOLD)):
            artoo.seeing_Vader = True
            print("Seeing Vader")
        else:
        	artoo.seeing_Vader = False

        #check if seeing Obi-Wan
        if ((int(classes[0][0]) == 2) and (scores[0][0] > THRESHOLD)):
            artoo.seeing_Obiwan = True
            print("Seeing Obi-Wan")
        else:
            artoo.seeing_Obiwan = False

        #check if seeing Leia
        if ((int(classes[0][0]) == 3) and (scores[0][0] > THRESHOLD)):
            artoo.seeing_Leia = True
            print("Seeing Leia")
        else:
        	artoo.seeing_Leia = False

        #update R2
        artoo.update(time1)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()
