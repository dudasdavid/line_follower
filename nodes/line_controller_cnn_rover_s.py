#!/usr/bin/env python

from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Twist
import rospy
import time

def controlRobot(data):
    global lastDirection
    direction = data.data[0]

    
    if direction == 0:
        print("forward")
        vel_msg.angular.z = 0
        vel_msg.linear.x = 0.32
    elif direction == 1:
        print("left")
        vel_msg.angular.z = -0.5
        vel_msg.linear.x = 0.32
        lastDirection = 1
    elif direction == 2:
        print("right")
        vel_msg.angular.z = 0.5
        vel_msg.linear.x = 0.32
        lastDirection = 2
    else:
        print("nothing")
        if lastDirection == 1:
            vel_msg.linear.x = -0.15
            vel_msg.angular.z = 0.5
        else:
            vel_msg.linear.x = -0.15
            vel_msg.angular.z = -0.5
        
        
    velocity_publisher.publish(vel_msg)

velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
vel_msg = Twist()
vel_msg.linear.x = 0
vel_msg.linear.y = 0
vel_msg.linear.z = 0
vel_msg.angular.x = 0
vel_msg.angular.y = 0
vel_msg.angular.z = 0

lastDirection = 1

rospy.init_node('line_follower', anonymous=True)
rospy.Subscriber('line_data', Int16MultiArray, controlRobot)

rospy.spin()