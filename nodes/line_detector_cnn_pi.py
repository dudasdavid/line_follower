#!/usr/bin/env python3
import threading
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16MultiArray
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import numpy as np
import math

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
import tensorflow as tf
import os, sys, imutils, argparse

withDisplay = False

x_width = 200 #200
x_offset = -4
y_height = 45
y_offset = 72
threshold_value = 200 # 60 for black line
binary_inverted = False

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


class DisplayThread(threading.Thread):
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
        self.image_pub = rospy.Publisher("/line_image/image_raw", Image, queue_size=1)
        # self.image_pub = rospy.Publisher("/line_image/image_raw/compressed", CompressedImage, queue_size=1)
        

    def run(self):
        if withDisplay:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("display", self.opencv_calibration_node.on_mouse)
        # cv2.createTrackbar("Camera type: \n 0 : pinhole \n 1 : fisheye", "display", 0,1, self.opencv_calibration_node.on_model_change)
        # cv2.createTrackbar("scale", "display", 0, 100, self.opencv_calibration_node.on_scale)
        
        while True:
            # print(self.queue.qsize())
            if self.queue.qsize() > 0:
                self.image = self.queue.get()
                processedImage = processImage(self.image, isDry = False)
                
                
                if withDisplay:
                    cv2.imshow("display", processedImage)
                
                try:
                    # msg = CompressedImage()
                    # msg.header.stamp = rospy.Time.now()
                    # msg.format = "jpeg"
                    # msg.data = np.array(cv2.imencode('.jpg', processedImage)[1]).tostring()
                    # Publish new image
                    # self.image_pub.publish(msg)
                
                    self.image_pub.publish(bridge.cv2_to_imgmsg(processedImage, "mono8"))
                except CvBridgeError as e:
                    print(e)
                
            else:
                time.sleep(0.01)
                
            k = cv2.waitKey(6) & 0xFF
            if k in [27, ord('q')]:
                rospy.signal_shutdown('Quit')

                

def queue_monocular(msg):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except CvBridgeError as e:
            print(e)
        else:
            q_mono.put(cv2_img)

def processImage(img, isDry = False):
    
    if isDry:
        return img
    
    
    start_time = time.clock()
    
    cropImg = img[int((240-y_height)/2 + y_offset):int((240+y_height)/2 + y_offset), int((320-x_width)/2 + x_offset):int((320+x_width)/2 + x_offset)]
    # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_GRAY2RGB)
    
    image = cv2.resize(cropImg, (28, 28))
    image = img_to_array(image)
    image = np.array(image, dtype="float") / 255.0
    image = image.reshape(-1, 28, 28, 1)
    
    with session.as_default():
        with session.graph.as_default():
            prediction = np.argmax(model.predict(image))
            
    array_to_send.data = [prediction]
    pubLine.publish(array_to_send) 

    print(time.clock()-start_time)
    
    return cropImg


pubLine = rospy.Publisher('line_data', Int16MultiArray, queue_size=1)
array_to_send = Int16MultiArray()

queue_size = 1      
q_mono = BufferQueue(queue_size)

display_thread = DisplayThread(q_mono)
display_thread.setDaemon(True)
display_thread.start()

bridge = CvBridge()

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

session = tf.Session(config=config)

set_session(session)


model = load_model("/home/pi/catkin_ws/src/line_follower/extras/model_pi")
model._make_predict_function()



rospy.init_node('image_listener')
# Define your image topic
image_topic = "/main_camera/image_raw"
# Set up your subscriber and define its callback
rospy.Subscriber(image_topic, Image, queue_monocular)
# Spin until ctrl + c
rospy.spin()
    
    

    

