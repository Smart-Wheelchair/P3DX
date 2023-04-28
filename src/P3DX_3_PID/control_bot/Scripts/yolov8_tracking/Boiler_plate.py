#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from my_track import process_images
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
import my_track
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = select_device('')
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
half = True

# Create as many strong sort instances as there are video sources
tracker_list = []
for i in range(1):
    tracker = create_tracker('deepocsort', './yolov8_tracking/trackers/deepocsort/configs/deepocsort.yaml', reid_weights, device, half)
    tracker_list.append(tracker, )
    if hasattr(tracker_list[i], 'model'):
        if hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()
outputs = [None]

def get_controls(x, z, Kp_l, Ki_l, Kd_l, Kp_a, Ki_a, Kd_a):
	global i_error_l 
	global i_error_a
	global d_error_l
	global d_error_a

	twist = Twist()

	p_error_l = z.astype(np.float32) - 1.5
	p_error_a = x - 640
	i_error_l += p_error_l
	i_error_a += p_error_a
	curr_d_error_l = p_error_l - d_error_l
	curr_d_error_a = p_error_a - d_error_a

	linear = Kp_l*p_error_l + Ki_l*i_error_l + Kd_l*curr_d_error_l
	angular = Kp_a*p_error_a + Ki_a*i_error_a + Kd_a*curr_d_error_a
	# print('linear: {} ,angular: {}  \n'.format(linear,angular))

	if linear > 0.2:
		linear = 0.2

	if angular > 0.2:
		angular = 0.2

	if linear < -0.2:
		linear = -0.2

	if angular < -0.2:
		angular = -0.2

	twist.linear.x = linear; twist.linear.y = 0; twist.linear.z = 0
	twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
	
	return twist

class FollowingNode:

    def __init__(self):
        rospy.init_node('following_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.cmd_vel_pub = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=1)
        self.target_depth_pub = rospy.Publisher('/target_depth', Float32, queue_size=1)
        kp = 0.5  # replace tuned values -Adi
        ki = 0.01 # replace tuned values -Adi
        kd = 0.2  # replace tuned values -Adi
        self.pid_controller = PIDController(kp=kp, ki=ki, kd=kd)
        

        self.cx = None
        self.cy = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply YOLO model to get the segmentation mask of the human
        im0, results = my_track.process_images(
            rgb_image=cv_image,
            tracker_list=tracker_list,
            outputs=outputs
            )
        
        cv2.imshow('Frame', im0)

        print(im0.shape)
        # print(results)

        #Publish Data to another node for Visualisation
        rospy.init_node('image_publisher', anonymous=True)
        pub = rospy.Publisher('image_topic', Image, queue_size=10)
        rate = rospy.Rate(10)  # 10 Hz
        bridge = CvBridge()
        while not rospy.is_shutdown():
            msg = bridge.cv2_to_imgmsg(im0, encoding='bgr8')  # convert image to ROS message
            pub.publish(msg)  # publish message on topic
            rate.sleep()

        if results is None or 1 not in results:
            print("NOBODY FOUND.")
        else:
            self.cx = int((results[1][0] + results[1][2]) / 2)
            self.cy = int((results[1][1] + results[1][3]) / 2)

        # Publish the target depth based on the depth image
        depth_image = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)

        # print('gottem')

        depth_array = np.array(self.bridge.imgmsg_to_cv2(depth_image), dtype=np.float32)
        target_depth = np.mean(depth_array[self.cy-4:self.cy + 4, self.cx-4:self.cx+4])
        self.cx = self.cx
        self.cy = self.cy
        self.target_depth_pub.publish(target_depth)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_array = np.array(depth_image, dtype=np.float32)
        # Compute the error between the target depth and the actual depth
        target_depth = rospy.wait_for_message('/target_depth', Float32)
        # error = target_depth.data - depth_array[self.cy, self.cx]
        # Compute the PID output

        target_depth = np.mean(depth_array[self.cy-4:self.cy + 4, self.cx-4:self.cx+4])
    
        print(self.cy, self.cx)

        twist = get_controls(self.cx, target_depth, 1/5, 0, 0.1,-1/500, 0, 0)

        self.cmd_vel_pub.publish(twist)

class PIDController:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def compute(self, error):
        self.error_sum += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        self.prev_error = error
        return output

if __name__ == '__main__':
    i_error_l = 0
    i_error_a = 0
    d_error_l = 0
    d_error_a = 0
    try:
        node = FollowingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass