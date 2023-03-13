#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from time import time

class VLMapsDataset:
    
    def __init__(self):
        self.odom_sub = message_filters.Subscriber("/RosAria/pose", Odometry)
        # self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.image_sub], 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)
        self.path = "/home/luke/Dev/RRC/smart-wheelchair/P3DX/data/vlmaps/indoor"
        self.counter = 0
    
    def callback(self, odom, image):
        pose = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ])

        print(f"Timestamp: {time()}, Pose: {pose}")

        np.save(os.path.join(self.path, f"pose_{self.counter}.npy"), pose)

        self.counter += 1

if __name__ == '__main__':
    rospy.init_node('listener')
    data = VLMapsDataset()
    rospy.spin()