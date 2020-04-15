#!/usr/bin/env python3

import threading
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rospy
from sensor_msgs.msg import Image

try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import numpy as np
import math

import os, sys, imutils, argparse

withDisplay = bool(int(rospy.get_param('~with_display', 0)))
withSave = bool(int(rospy.get_param('~with_save', 1)))

isMono = bool(int(rospy.get_param('~mono', 1)))
saveTime = float(rospy.get_param('~save_time', 0.5))

x_width = int(rospy.get_param('~x_width', 200))
x_offset = int(rospy.get_param('~x_offset', 0))
y_height = int(rospy.get_param('~y_height', 45))
y_offset = int(rospy.get_param('~y_offset', 72))

class BufferQueue(Queue):
    """Slight modification of the standard Queue that discards the oldest item
    when adding an item and the queue is full.
    """
    def put(self, item, *args, **kwargs):
        # The base implementation, for reference:
        # https://github.com/python/cpython/blob/2.7/Lib/Queue.py#L107
        # https://github.com/python/cpython/blob/3.8/Lib/queue.py#L121
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class ImageThread(threading.Thread):
    """
    Thread that displays the current images
    It is its own thread so that all display can be done
    in one thread to overcome imshow limitations and
    https://github.com/ros-perception/image_pipeline/issues/85
    """
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.image = None
        

    def run(self):
        if withDisplay:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)
        
        imageIndex = 0
        path = rospy.get_param('~save_path')  ## this should be a parameter
        # path = "/home/david/Pictures/saves/"  ## this should be a parameter
        while True:
            if self.queue.qsize() > 0:
                self.image = self.queue.get()
                processedImage = processImage(self.image, isDry = False)

                if withSave:
                    cv2.imwrite(path + str(imageIndex) + ".jpg", processedImage)
                    print("file saved")
                    imageIndex+=1
                    time.sleep(saveTime)
                
                if withDisplay:
                    cv2.imshow("display", processedImage)
                
            else:
                time.sleep(0.01)
                
            k = cv2.waitKey(6) & 0xFF
            if k in [27, ord('q')]:
                rospy.signal_shutdown('Quit')

                

def queue_monocular(msg):
        try:
            # Convert your ROS Image message to OpenCV2
            if isMono:
                cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            else:
                cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            q_mono.put(cv2_img)

def processImage(img, isDry = False):
    
    if isDry:
        return img
    
    start_time = time.clock()

    height, width = img.shape[:2]

    if height != 240 or width != 320:
        dim = (320, 240)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cropImg = img[int((240-y_height)/2 + y_offset):int((240+y_height)/2 + y_offset), int((320-x_width)/2 + x_offset):int((320+x_width)/2 + x_offset)]
    # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2RGB)
    
    return cropImg
    

queue_size = 1
q_mono = BufferQueue(queue_size)

display_thread = ImageThread(q_mono)
display_thread.setDaemon(True)
display_thread.start()

bridge = CvBridge()


rospy.init_node('image_listener')
# Define your image topic
# image_topic = "/main_camera/image_raw"
image_topic = rospy.get_param('~image_topic')
# Set up your subscriber and define its callback
rospy.Subscriber(image_topic, Image, queue_monocular)
# Spin until ctrl + c
rospy.spin()
    
    

    

