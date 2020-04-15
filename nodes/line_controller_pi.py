#!/usr/bin/env python

from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Twist
import rospy

P = 0.005

def controlRobot(data):
    xmax = data.data[0]
    valid = data.data[1]
    xval = data.data[2]
    angle = data.data[3]
    
    if valid:
        vel_msg.angular.z = P * -(xval - xmax/2)
        
        vel_msg.linear.x = 0.15
    else:
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        
    velocity_publisher.publish(vel_msg)

velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
vel_msg = Twist()
vel_msg.linear.x = 0
vel_msg.linear.y = 0
vel_msg.linear.z = 0
vel_msg.angular.x = 0
vel_msg.angular.y = 0
vel_msg.angular.z = 0

rospy.init_node('line_follower', anonymous=True)
rospy.Subscriber('line_data', Int16MultiArray, controlRobot)

rospy.spin()