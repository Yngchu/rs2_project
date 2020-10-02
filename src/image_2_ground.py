#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bridge = CvBridge()
info_matrix = Int16MultiArray()
# Duckiebot parameters
Duckie_width = (13 + 17) * 0.5  # centimeter
# Camera parameters
fx = 317.013
px = 305.586
fy = 317.651
py = 233.604
k_matrix = [[fx, 0, px],
			[0, fy, py],
			[0, 0, 1]]
inv_k = np.linalg.inv(k_matrix)
ave_f = (fx + fy) * 0.5


def coordinates_callback(MultiArray):
	info_m = np.array(MultiArray.data)
	if len(info_m) > 0:
		target_width = info_m[2]
		target_height = info_m[3]
		target_centroid = [info_m[0], info_m[1]]
		image_2_ground(target_width, target_centroid)


def image_2_ground(width, centroid):
	distance = (Duckie_width * ave_f) / width
	target = [[centroid[0] * distance], [centroid[1] * distance], [distance]]
	x_cam = np.matmul(inv_k, target)
	pose_matrix.data = [x_cam[0][0], distance]
	pub_duckie_pose.publish(pose_matrix)
	# print (x_cam[0][0], distance)
	# plt.scatter(x_cam[0][0], distance)
	# plt.show()
def show_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(1)


if __name__ == '__main__':
	rospy.init_node('ground_pose_collector')
	rospy.loginfo('duckie_2_ground node started')
	pose_matrix = Int16MultiArray()
	while not rospy.is_shutdown():
		try:
			sub_coordinates = rospy.Subscriber('target_coordinates', Int16MultiArray, coordinates_callback)
			pub_duckie_pose = rospy.Publisher('duckie_image_2_ground', Int16MultiArray, queue_size=10)
			rospy.spin()
		except rospy.ROSInterruptException:
			pass
