"""
trap_detector_node.py

ROS 2 node for Stage 0 of the pest detection pipeline.
Subscribes to camera images, detects yellow sticky traps using HSV
color filtering, crops the trap region, and publishes the cropped
image for downstream classification nodes.

Topics:
    Subscribes : /camera/image  (sensor_msgs/msg/Image)
    Publishes  : /trap/cropped_image  (sensor_msgs/msg/Image)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
import numpy as np
from cv_bridge import CvBridge

YELLOW_LOWER = np.array([15, 20, 80])
YELLOW_UPPER = np.array([40, 255, 255])
MIN_TRAP_AREA = 5000
MAX_TRAP_AREA = 2000000


class TrapDetectorNode(Node):
    """
    ROS 2 node that detects yellow sticky traps in camera images.

    Purpose:
        Converts the standalone trap_detector.py script into a ROS node
        that fits into the Amiga's real-time processing pipeline. Instead
        of reading from disk, it receives live camera frames via a ROS
        topic and publishes cropped trap images for the binary filter node.

    Subscribes:
        /camera/image (sensor_msgs/msg/Image): Raw camera frames from
            the Amiga's Oak-D camera.

    Publishes:
        /trap/cropped_image (sensor_msgs/msg/Image): Cropped trap region,
            only published when a trap is successfully detected.
        /trap/detected (std_msgs/msg/Bool): True if trap found, False if not.
    """

    def __init__(self):
        super().__init__('trap_detector_node')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.crop_publisher = self.create_publisher(
            Image,
            '/trap/cropped_image',
            10
        )

        self.detected_publisher = self.create_publisher(
            Bool,
            '/trap/detected',
            10
        )

        self.get_logger().info('Trap detector node started, listening on /camera/image')

    def detect_trap(self, image):
        """
        Detect and crop the yellow sticky trap from a camera frame.

        Purpose:
            Core trap detection logic adapted from trap_detector.py for
            use inside a ROS callback. Takes a numpy image array and
            returns the cropped trap region using HSV color filtering
            and contour detection.

        Args:
            image (np.ndarray): BGR image array from the camera.

        Returns:
            tuple: (cropped, bbox) where cropped is the trap region as
                   np.ndarray or None, and bbox is (x,y,w,h) or None.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        valid = [c for c in contours if MIN_TRAP_AREA < cv2.contourArea(c) < MAX_TRAP_AREA]

        if not valid:
            largest_area = max(cv2.contourArea(c) for c in contours)
            img_area = image.shape[0] * image.shape[1]
            if largest_area > img_area * 0.3:
                return image, (0, 0, image.shape[1], image.shape[0])
            return None, None

        largest = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return image[y:y+h, x:x+w], (x, y, w, h)

    def image_callback(self, msg):
        """
        Callback that runs every time a new camera frame arrives.

        Purpose:
            Converts the ROS Image message to a numpy array, runs trap
            detection, and publishes the cropped trap image if found.
            This is the core ROS integration point where the pipeline
            becomes event-driven rather than file-based.

        Args:
            msg (sensor_msgs/msg/Image): Incoming camera frame from
                the /camera/image topic.

        Returns:
            None
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cropped, bbox = self.detect_trap(image)

        detected_msg = Bool()
        detected_msg.data = cropped is not None
        self.detected_publisher.publish(detected_msg)

        if cropped is not None:
            crop_msg = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
            self.crop_publisher.publish(crop_msg)
            self.get_logger().info(f'Trap detected at {bbox}, published to /trap/cropped_image')
        else:
            self.get_logger().debug('No trap detected in this frame')


def main(args=None):
    rclpy.init(args=args)
    node = TrapDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
