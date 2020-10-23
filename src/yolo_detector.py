#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
# from darknet_ros_msgs.msg import BoundingBox
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

bridge = CvBridge()
info_matrix = Int16MultiArray()

weight = "/home/yingchu/catkin_ws/src/rs2_project/src/tiny-yolo-duckie-3.weights"
cfg = "/home/yingchu/catkin_ws/src/rs2_project/src/yolov3-tiny-duckie-3.cfg"
# Load Yolo
net = cv2.dnn.readNet(weight, cfg)


classes = ["Duckiebot"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 2))

# Initialize Twist publisher
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
curr_orientation_angle = 0.0

def process_image(msg):
	try:
		# Convert compressed_image -> opencv image
		np_arr = np.fromstring(msg.data, np.uint8)
		ori_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		height, width, channels = ori_img.shape
		# Detecting objects
		blob = cv2.dnn.blobFromImage(ori_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)
		# Showing information on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)
		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
		# font = cv2.FONT_HERSHEY_PLAIN
		if len(boxes) > 0:
			for i in range(len(boxes)):
				if i in indexes:
					x, y, w, h = boxes[0]
					# label = str(classes[class_id[i]])
					# confidence = confidence[i]
					# color = colors[class_ids[i]]
					# cv2.rectangle(ori_img, (x, y), (x + w, y + h), (255,0,0), 2)
					info_matrix.data = [x + w/2, y + h/2, w, h]
					pub_coordinates.publish(info_matrix)
				# cv2.putText(ori_img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (0,0,0), 3)
		# # show_image(ori_img)
		# ros_img = bridge.cv2_to_imgmsg(ori_img, "bgr8")
		# pub_yolo_image.publish(ros_img)
	except Exception as err:
		print err


def show_image(img):
	cv2.imshow('Image', img)
	cv2.waitKey(1)


if __name__ == '__main__':
	rospy.init_node('yolo_detector')
	rospy.loginfo('yolo_detector node started')
	while not rospy.is_shutdown():
		try:
			rospy.Subscriber("/duckiebot01/camera_node/image/compressed", CompressedImage, process_image)
			# rospy.Subscriber("/duckiebot/camera_node/image/raw", Image, process_image)
			pub_coordinates = rospy.Publisher('target_coordinates', Int16MultiArray, queue_size=10)
			# pub_yolo_image = rospy.Publisher('yolo_detector', Image, queue_size=10)
			rospy.spin()
		except rospy.ROSInterruptException:
			pass
