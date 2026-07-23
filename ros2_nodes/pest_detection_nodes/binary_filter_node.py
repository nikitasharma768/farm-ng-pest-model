"""
binary_filter_node.py

ROS 2 node for Stage A of the pest detection pipeline.
Subscribes to cropped trap images, runs the binary insect/not-insect
classifier, and publishes results for the species classifier node.

Topics:
    Subscribes : /trap/cropped_image  (sensor_msgs/msg/Image)
    Publishes  : /insect/detected     (std_msgs/msg/Bool)
                 /insect/image        (sensor_msgs/msg/Image)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import tempfile
import os


class BinaryFilterNode(Node):
    """
    ROS 2 node that classifies cropped trap images as insect or not-insect.

    Purpose:
        Converts the binary classifier stage from pipeline.py into a ROS
        node. Receives cropped trap images, runs the YOLOv8 binary
        classifier, and forwards confirmed insect images to the species
        classifier node via a separate topic.

    Subscribes:
        /trap/cropped_image (sensor_msgs/msg/Image): Cropped trap region
            from the trap detector node.

    Publishes:
        /insect/detected (std_msgs/msg/Bool): True if insect found.
        /insect/image (sensor_msgs/msg/Image): The crop image, only
            published when an insect is confirmed present.
    """

    def __init__(self):
        super().__init__('binary_filter_node')

        self.declare_parameter('model_path',
            '/home/nikit/farm-ng-pest-model/models/checkpoints/binary_insect_classifier/weights/best.pt')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        self.get_logger().info(f'Loading binary classifier from {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/trap/cropped_image',
            self.image_callback,
            10
        )

        self.detected_publisher = self.create_publisher(Bool, '/insect/detected', 10)
        self.image_publisher = self.create_publisher(Image, '/insect/image', 10)

        self.get_logger().info('Binary filter node ready, listening on /trap/cropped_image')

    def image_callback(self, msg):
        """
        Callback that classifies each received trap image as insect or not.

        Purpose:
            Saves the image temporarily to disk (required by YOLOv8's predict
            method), runs inference, publishes the result, and if an insect
            is detected, forwards the image to the species classifier topic.

        Args:
            msg (sensor_msgs/msg/Image): Cropped trap image from the
                trap detector node.

        Returns:
            None
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        result = self.model.predict(tmp_path, verbose=False)[0]
        os.unlink(tmp_path)

        pred_idx = result.probs.top1
        pred_label = result.names[pred_idx]
        pred_conf = float(result.probs.top1conf)

        is_insect = pred_label == 'insect' and pred_conf >= 0.5

        detected_msg = Bool()
        detected_msg.data = is_insect
        self.detected_publisher.publish(detected_msg)

        if is_insect:
            self.image_publisher.publish(msg)
            self.get_logger().info(f'Insect detected (conf {pred_conf:.2f}), forwarding to species classifier')
        else:
            self.get_logger().info(f'No insect (conf {pred_conf:.2f}), discarding')


def main(args=None):
    rclpy.init(args=args)
    node = BinaryFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
