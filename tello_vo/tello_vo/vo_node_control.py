#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist    
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np
import matplotlib.pyplot as plt

class VONode(Node):
    def __init__(self):
        super().__init__('vo_node')
        # Path publisher
        self.frame_count = 0
        self.path_pub = self.create_publisher(Path, '/vo_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        self.vx_body, self.vy_body = 0.0, 0.0
        # Camera Calibration
        fx, fy = ____, ____
        cx, cy = ____, ____
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)

        # ORB + FLANN
        #The number of keypoints can be set to 1500, 3000, or other values—you can adjust this parameter as needed to suit your application.
        self.orb = ____ 
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params,
                                           searchParams=search_params)

        # CvBridge
        self.bridge = CvBridge()
        self.last_gray = None
        self.cur_pose = np.eye(4, dtype=np.float64)
        self.x_vals, self.z_vals = [], []

        # Matplotlib interactive
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_title("Camera Trajectory (x vs z)")
        self.ax.set_xlabel("x(m)")
        self.ax.set_ylabel("z(m)")
        self.line, = self.ax.plot([], [], 'b.-')
        self.ax.grid(True)
        self.ax.axis('equal')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.current_vel    = 0.0
        self.cmd_timestamp  = None
        self.prev_img_stamp = None
        
        self.create_subscription(
            Image, '/image_raw',
            self.image_callback,
            qos
        )
        self.get_logger().info("VO node started, waiting for /image_raw ...")

        self.create_subscription(
            Twist, '/cmd_vel',
            self.cmd_callback, 10)

        self.get_logger().info("VO node ready (image + cmd_vel)")

    def cmd_callback(self, msg: Twist):
        self.vx_body = msg.linear.x
        self.vy_body = msg.linear.y
        self.cmd_timestamp = self.get_clock().now().nanoseconds * 1e-9
        
    def image_callback(self, msg: Image):
        
        '''
        During the first 20 frames after image input begins, the program only collects images for initialization and does not perform pose estimation.
        self.frame_count += 1
        if ____:
            ____
        '''
        
        img_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if self.prev_img_stamp is None:
            self.prev_img_stamp = img_stamp
        
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.last_gray is None:
            self.last_gray = gray
            return

        kp1, des1 = self.orb.detectAndCompute(self.last_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None:
            self.last_gray = gray
            return

        # 4. FLANN + ratio test
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m1, m2 = pair
            if m1.distance < 0.8 * m2.distance:
                good.append(m1)
        if len(good) < 8:
            self.last_gray = gray
            return
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # 5. Essential + recoverPose
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K,
                                     cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        
        dt = img_stamp - self.prev_img_stamp
        self.prev_img_stamp = img_stamp
        
        raw_t     = t.reshape(3)
        norm_t    = np.linalg.norm(raw_t) + 1e-9
        v_cam     = np.array([-self.vy_body, 0.0, self.vx_body])
        raw_t_unit = raw_t / norm_t
        v_along_t  = np.dot(v_cam, raw_t_unit)
        d_cmd      = v_along_t * dt
        scale      = d_cmd / norm_t
        t_scaled   = raw_t * scale
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = t_scaled
        self.cur_pose = self.cur_pose @ T

        x, z = self.cur_pose[0,3], self.cur_pose[2,3]
        self.x_vals.append(x)
        self.z_vals.append(z)
        self.line.set_data(self.x_vals, self.z_vals)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        
        # PoseStamped，Push to Path 
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = 'odom'
        ps.pose.position.x = x
        ps.pose.position.y = self.cur_pose[1,3]
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0

        self.path_msg.poses.append(ps)
        self.path_msg.header.stamp = msg.header.stamp
        self.path_pub.publish(self.path_msg)
        
        self.last_gray = gray

def main(args=None):
    rclpy.init(args=args)
    node = VONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.fig.savefig('final_trajectory.png')
        node.destroy_node()
        rclpy.shutdown()
        plt.close(node.fig)

if __name__ == '__main__':
    main()

