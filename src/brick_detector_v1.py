#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import CompressedImage
#from std_msgs.msg import Int16MultiArray
import cv2
import math
import numpy as np
import time
#import imutils
bridge = CvBridge()
def nothing(x):
    pass

vel_pub=rospy.Publisher('/cmd_vel', Twist, queue_size=1)
curr_orientation_angle=0.0
overridden_pub = rospy.Publisher('overridden', Bool, queue_size=10)



def start_node():
    global ranges
    rospy.init_node('brick_detector')
    rospy.loginfo('brick_detector node started')
    # rospy.Subscriber("/camera/depth/image_raw", Image, depth_image)
    rospy.Subscriber("/duckiebot01/camera_node/image/compressed", CompressedImage, process_image)
    # rospy.Subscriber("/scan", LaserScan, read_distance)
    rospy.spin()

def depth_image(msg):
    global resize_depth
    depth_orig = bridge.imgmsg_to_cv2(msg,"passthrough")
    resize_depth = cv2.resize(depth_orig,(640,360))
    # cv2.imshow('Depth',resize_depth)

def read_distance(msg):
    global ranges

    ranges = msg.ranges

def process_image(msg):
    global ranges
    try:
       # convert sensor_msgs/Image to OpenCV Image
        np_arr = np.fromstring(msg.data, np.uint8)
        orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # resize_orig = cv2.resize(orig,(640,360))
        brick_filtered = detect_brick(orig)
        # show_image(resize_orig)
        # targets = detect_corner(resize_orig, brick_filtered)
        
    except Exception as err:
        print err

def detect_brick(frame):
    # filter for blue lane lines
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar('LH', "Trackbars", 0, 179, nothing)
    cv2.createTrackbar('LS', "Trackbars", 0, 255, nothing)
    cv2.createTrackbar('LV', "Trackbars", 0, 255, nothing)
    cv2.createTrackbar('UH', "Trackbars", 179, 179, nothing)
    cv2.createTrackbar('US', "Trackbars", 255, 255, nothing)
    cv2.createTrackbar('UV', "Trackbars", 255, 255, nothing)
    l_h = cv2.getTrackbarPos('LH', "Trackbars")
    l_s = cv2.getTrackbarPos('LS', "Trackbars")
    l_v = cv2.getTrackbarPos('LV', "Trackbars")
    u_h = cv2.getTrackbarPos('UH', "Trackbars")
    u_s = cv2.getTrackbarPos('US', "Trackbars")
    u_v = cv2.getTrackbarPos('UV', "Trackbars")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result1 = cv2.bitwise_and(frame, frame, mask=mask)
    result = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    show_image(result1)
    return result

def detect_corner(img_orin, img_gray):
    global central_x, central_y
    x_list = []
    y_list = []
    depth_list=[]
    corners = cv2.goodFeaturesToTrack(img_gray, 10, 0.01, 0.1)
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            x_list.append(x)
            y_list.append(y)
            # cv2.circle(img_orin, (x, y), 3, 255, -1)
        central_x = (np.min(x_list)+np.max(x_list)) // 2
        controller(central_x)
        central_y = (np.min(y_list)+np.max(y_list)) // 2
        centroid = (central_x, central_y)
        # print resize_depth[central_x, central_y]
        cv2.circle(img_orin,centroid, 3,(255,255,0),1)
        bound_x1 = np.min(x_list)
        bound_y1 = np.min(y_list)
        bound_x2 = np.max(x_list)
        bound_y2 = np.max(y_list)
        # roi_depth = resize_depth[bound_x1:bound_x2, bound_y1:bound_y2]
        # # if len(roi_depth)>0:
        # #     raw = len(roi_depth)
        # #     col = len(roi_depth[0])
        # #     for i in range(col):
        # #         for j in range(raw):
        # #             distance = roi_depth[j, i]
        # #             if distance > 0:
        # #                 depth_list.append(distance)
        # print roi_depth
        bounding_box = cv2.rectangle(img_orin, (bound_x1, bound_y1),(bound_x2,bound_y2),(255,0,0),2)
        # return central_x
    show_image(img_orin)
    # cv2.imshow('Figure2', img_orin)



def controller(orientation):
    global ranges
    # print np.shape(ranges)
    valid_ranges = []

    if orientation is not None:
        # print "Detection."
        range_position = int((np.interp(orientation,[0,640],[35,-35])))#+360)%360)

        # print (range_position+1, range_position, range_position-1)
        measurements = [ranges[range_position+1], ranges[range_position], ranges[range_position-1]]
        for measurement in measurements:
            if measurement <= 3.5:
                valid_ranges.append(measurement)
        # print measurements
        # print valid_ranges
        # print range_sum
        if len(valid_ranges) != 0:
            actual_range = sum(valid_ranges)/len(valid_ranges)
        else:
            actual_range = 3.5
        # print ("Forward range: %s" % actual_range)
        # if range_position < 0:
        #     range_position += 359
        # print actual_range
        # forward_range = ranges[range_position]
        # print forward_range
        
        angular_vel = np.interp(orientation,[0,640],[0.2,-0.2])
        if abs(320-orientation) < 30:
            # print abs(960-orientation)
            forward_vel = np.interp(actual_range,[0.5,3.5],[0.0,0.4])
        else:
            forward_vel = 0.00
        overridden_pub.publish(Bool(data=True))
        publishing_vel(forward_vel, angular_vel)
    else:
        # print "No detection."
        forward_vel = 0.00
        angular_vel = 0.00
        overridden_pub.publish(Bool(data=False))
        publishing_vel(forward_vel, angular_vel)
    # print ("Forward velocity: %s, Angular velocity: %s" % (forward_vel, angular_vel))
    # print ""

def publishing_vel(forward_vel, angular_vel):
    vel = Twist()
    vel.angular.x = 0.0
    vel.angular.y = 0.0
    vel.angular.z = angular_vel
    vel.linear.x = forward_vel
    vel.linear.y = 0.0
    vel.linear.z = 0.0
    vel_pub.publish(vel)

def show_image(img):
    cv2.imshow('Figure', img)
    cv2.waitKey(1)

if __name__ == '__main__':
    global ranges
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
