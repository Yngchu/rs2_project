#!/usr/bin/env python
import rospy
# from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
# from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
# from darknet_ros_msgs.msg import BoundingBox
import cv2
import numpy as np
import csv
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# with open("/home/yingchu/catkin_ws/src/rs2_project/src/trajectory.csv", 'wb') as f:
# 	csv_write = csv.writer(f)
# 	csv_head = ["time", "x", "y"]
# 	csv_write.writerow(csv_head)
global l_distance
l_distance = 0
global l_x_cam
l_x_cam = 0
path = "/csv_file/trajectory2.csv"
# pub_duckie_pose = rospy.Publisher('duckie_image_2_ground', Int16MultiArray, queue_size=10)
# info_matrix = BoundingBox()
# Duckiebot parameters
Duckie_width = (130 + 170) * 0.5  # millimeter
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

def write_csv(target_x,target_y):
	with open(path,'a+') as f:
		csv_write_a = csv.writer(f)
		data_row = [target_x, target_y]
		csv_write_a.writerow(data_row)

def coordinates_callback(MultiArray):
	info_m = np.array(MultiArray.data)
	if len(info_m) > 0:
		target_width = info_m[2]
		target_height = info_m[3]
		target_centroid = [info_m[0], info_m[1]]
		image_2_ground(target_width, target_centroid)


def image_2_ground(width, centroid):
	global l_distance 
	global l_x_cam
	distance = (Duckie_width * ave_f) / width
	target = [[centroid[0] * distance], [centroid[1] * distance], [distance]]
	x_cam = np.matmul(inv_k, target)
	if l_distance != distance or l_x_cam != x_cam[0][0]:
		write_csv(x_cam[0][0], distance)
		l_distance = distance
		l_x_cam = x_cam[0][0]
		rospy.loginfo('Done')

	# write_csv(x_cam[0][0])
	# print distance
	# rospy.sleep(0.5)
	# pose_matrix.data = [x_cam[0][0], distance]
	# print distance
	# pub_duckie_pose.publish(pose_matrix)
	# print (x_cam[0][0], distance)
	# plt.scatter(x_cam[0][0], distance)
	# plt.show()

def show_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(1)



if __name__ == '__main__':
	rospy.init_node('ground_pose_collector')
	rospy.loginfo('duckie_2_ground node started')
	rate = rospy.Rate(2)
	with open(path, 'wb') as f:
		csv_write = csv.writer(f)
		csv_head = ["x","y"]
	rospy.loginfo('csv starting')
	# pose_matrix = Int16MultiArray()
	while not rospy.is_shutdown():
		try:
			sub_coordinates = rospy.Subscriber('target_coordinates', Int16MultiArray, coordinates_callback)
			rate.sleep()
		except rospy.ROSInterruptException:
			pass
