#!/usr/bin/env python

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

withDisplay = False

x_width = 128
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
        self.maskImage_pub = rospy.Publisher("/line_image/mask_raw", Image, queue_size=1)
        # self.image_pub = rospy.Publisher("/line_image/image_raw/compressed", CompressedImage, queue_size=1)
        

    def run(self):
        if withDisplay:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("display", self.opencv_calibration_node.on_mouse)
        # cv2.createTrackbar("Camera type: \n 0 : pinhole \n 1 : fisheye", "display", 0,1, self.opencv_calibration_node.on_model_change)
        # cv2.createTrackbar("scale", "display", 0, 100, self.opencv_calibration_node.on_scale)
        
        while True:
            # print(self.queue.qsize())
            if self.queue.qsize() > 0:
                self.image = self.queue.get()
                processedImage, maskImage = processImage(self.image, isDry = False)
                if withDisplay:
                    cv2.imshow("display", processedImage)
                    cv2.imshow("mask", maskImage)
                
                try:
                    # msg = CompressedImage()
                    # msg.header.stamp = rospy.Time.now()
                    # msg.format = "jpeg"
                    # msg.data = np.array(cv2.imencode('.jpg', processedImage)[1]).tostring()
                    # Publish new image
                    # self.image_pub.publish(msg)
                
                    self.image_pub.publish(bridge.cv2_to_imgmsg(processedImage, "bgr8"))
                    self.maskImage_pub.publish(bridge.cv2_to_imgmsg(maskImage, "bgr8"))
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
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    cropImg = img[int((240-y_height)/2 + y_offset):int((240+y_height)/2 + y_offset), int((320-x_width)/2 + x_offset):int((320+x_width)/2 + x_offset)]
    monoImg = cv2.cvtColor(cropImg, cv2.COLOR_RGB2GRAY)
    
    rows,cols = cropImg.shape[:2]
    
    # Gaussian blur
    blurImg = cv2.GaussianBlur(monoImg,(5,5),0)
    # Color thresholding
    if binary_inverted:
        ret,threshImg = cv2.threshold(blurImg,threshold_value,255,cv2.THRESH_BINARY_INV)
    else:
        ret,threshImg = cv2.threshold(blurImg,threshold_value,255,cv2.THRESH_BINARY)
        
    # Erode and dilate to remove accidental line detections
    mask = cv2.erode(threshImg, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Find the contours of the frame
    contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
    
    
    # Find the biggest contour (if detected)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.line(cropImg,(cx,0),(cx,rows),(255,0,0),1)
        cv2.line(cropImg,(0,cy),(cols,cy),(255,0,0),1)
        cv2.drawContours(cropImg, contours, -1, (0,255,0), 1)
    
    
        
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((cols-x)*vy/vx)+y)
        
        angle = math.atan2(vy,vx)
        if angle < 0:
            angle = angle + math.pi / 2
        else:
            angle = angle - math.pi / 2
        
        angle *= -1
        
        # print(math.degrees(angle))
        # cv2.line(cropImg,(cols-1,righty),(0,lefty),(0,0,255),1)
        cv2.line(cropImg, (int(cx-math.cos(angle - math.pi / 2)*100), int(cy+math.sin(angle - math.pi / 2 )*100)), (int(cx+math.cos(angle - math.pi / 2)*100), int(cy-math.sin(angle - math.pi / 2 )*100)), (0,0,255), 1)
            
        array_to_send.data = [cols, 1, cx, angle]
    
        
    else:
        array_to_send.data = [cols, 0, 0, 0]
    
    pubLine.publish(array_to_send)   
    
    maskImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return cropImg, maskImage


pubLine = rospy.Publisher('line_data', Int16MultiArray, queue_size=1)
array_to_send = Int16MultiArray()

queue_size = 1      
q_mono = BufferQueue(queue_size)

display_thread = DisplayThread(q_mono)
display_thread.setDaemon(True)
display_thread.start()

bridge = CvBridge()

rospy.init_node('image_listener')
# Define your image topic
image_topic = "/main_camera/image_raw"
# Set up your subscriber and define its callback
rospy.Subscriber(image_topic, Image, queue_monocular)
# Spin until ctrl + c
rospy.spin()
    
    

    

