#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
# from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
# from darknet_ros_msgs.msg import BoundingBox
from tensorflow.keras.models import load_model
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
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

global theta
theta = 0
global l_distance
l_distance = 0
global l_x_cam
l_x_cam = 0
global target_predict
target_predict = []
global model
model = load_model('/home/yingchu/catkin_ws/src/rs2_project/src/10step_lstm_model.h5')
model.summary()
# path = "/csv_file/trajectory2.csv"
# pub_duckie_pose = rospy.Publisher('duckie_image_2_ground', Int16MultiArray, queue_size=10)
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


def coordinates_callback(MultiArray):
	info_m = np.array(MultiArray.data)
	if len(info_m) > 0:
		target_width = info_m[2]
		target_height = info_m[3]
		target_centroid = [info_m[0], info_m[1]]
		image_2_ground(target_width, target_centroid)


def image_2_ground(width, centroid):
	global target_predict
	global l_distance
	global l_x_cam
	global theta
	distance = (Duckie_width * ave_f) / width
	target = [[centroid[0] * distance], [centroid[1] * distance], [distance]]
	x_cam = np.matmul(inv_k, target)
	if l_distance != distance or l_x_cam != x_cam[0][0]:
		data = [x_cam[0][0]/100, distance/100]
		target_predict.append(data)
		l_distance = distance/100
		l_x_cam = x_cam[0][0]/100
		
		if len(target_predict) == 10:
			rospy.loginfo('Input Trajectory')
			target_predict_l = target_predict
			target_predict_l = np.array(target_predict_l, dtype=float)
			target_predict_l = target_predict_l.reshape(1, 10, 2)
			# print target_predict_l
			results = model.predict(target_predict_l)
			results = results.reshape(10, 2)
			rito = (results[9][0] - results[0][0])/(results[9][1] - results[0][1])
			theta = np.arctan(rito) * 180 / np.pi
			rospy.loginfo("predicted theta %s", theta)
			target_predict = []
			if -20 < theta < 20:
				publishing_vel(0.3, 0.3)
			elif theta <= -20:
				publishing_vel(0.3, -0.3)
			else:
				publishing_vel(0.3, 0.1)



def show_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(1)


def publishing_vel(forward_vel, angular_vel):
	vel = Twist()
	vel.angular.x = 0.0
	vel.angular.y = 0.0
	vel.angular.z = angular_vel
	vel.linear.x = forward_vel
	vel.linear.y = 0.0
	vel.linear.z = 0.0
	vel_pub.publish(vel)


if __name__ == '__main__':
	rospy.init_node('predictor')
	rospy.loginfo('prediction node started')
	publishing_vel(0, 0)
	rate = rospy.Rate(2)
	while not rospy.is_shutdown():
		try:
			sub_coordinates = rospy.Subscriber('target_coordinates', Int16MultiArray, coordinates_callback)
			rate.sleep()
		except rospy.ROSInterruptException:
			pass
