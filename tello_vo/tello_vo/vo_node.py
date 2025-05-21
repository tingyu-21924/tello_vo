#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
import matplotlib.pyplot as plt

class VONode(Node):
    def __init__(self):
        super().__init__('vo_node')

        #Camera Calibration
        fx, fy = 1120.6219, 1133.0317
        cx, cy = 357.7350, 640.5663
        #fx, fy = ?, ?
        #cx, cy = ?, ?
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)

        # ORB + FLANN
        self.orb = cv2.ORB_create(3000)
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
        self.create_subscription(
            Image, '/image_raw',
            self.image_callback,
            qos
        )
        self.get_logger().info("VO node started, waiting for /image_raw ...")

    def image_callback(self, msg: Image):
        # 1. ROS Image → BGR → 灰階
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
        good = [m for m,n in matches if m.distance < 0.8*n.distance]
        if len(good) < 8:
            self.last_gray = gray
            return

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # 5. Essential + recoverPose
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K,
                                     cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3], T[:3, 3] = R, (t.reshape(3) / 100.0)
        self.cur_pose = self.cur_pose @ np.linalg.inv(T)

        x, z = self.cur_pose[0,3], self.cur_pose[2,3]
        self.x_vals.append(x)
        self.z_vals.append(z)

        self.line.set_data(self.x_vals, self.z_vals)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.last_gray = gray

def main(args=None):
    rclpy.init(args=args)
    node = VONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close(node.fig)

if __name__ == '__main__':
    main()

