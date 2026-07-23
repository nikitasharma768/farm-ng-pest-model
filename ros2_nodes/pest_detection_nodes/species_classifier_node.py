"""
species_classifier_node.py

ROS 2 node for Stage B of the pest detection pipeline.
Subscribes to confirmed insect images, runs the 102-species YOLOv8
classifier, and publishes the species result for the heatmap node.

Topics:
    Subscribes : /insect/image   (sensor_msgs/msg/Image)
    Publishes  : /insect/species (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import tempfile
import os
import json


class SpeciesClassifierNode(Node):
    """
    ROS 2 node that identifies the pest species from confirmed insect images.

    Purpose:
        Converts Stage B of pipeline.py into a ROS node. Receives images
        that have already passed the binary insect filter, runs the 102-class
        YOLOv8 species classifier, and publishes a JSON string containing
        the species name, confidence, and top-5 predictions.

    Subscribes:
        /insect/image (sensor_msgs/msg/Image): Confirmed insect images
            from the binary filter node.

    Publishes:
        /insect/species (std_msgs/msg/String): JSON string with species
            name, confidence, and top-5 predictions.
    """

    def __init__(self):
        super().__init__('species_classifier_node')

        self.declare_parameter('model_path',
            '/home/nikit/farm-ng-pest-model/models/checkpoints/best.pt')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        self.get_logger().info(f'Loading species classifier from {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/insect/image',
            self.image_callback,
            10
        )

        self.species_publisher = self.create_publisher(String, '/insect/species', 10)

        self.get_logger().info('Species classifier node ready, listening on /insect/image')

    def image_callback(self, msg):
        """
        Callback that classifies each confirmed insect image to species level.

        Purpose:
            Runs the 102-class YOLOv8 model on the received image and
            publishes the top-1 species prediction along with top-5
            alternatives as a JSON string. This allows the heatmap node
            to log full prediction details per trap detection event.

        Args:
            msg (sensor_msgs/msg/Image): Confirmed insect image from
                the binary filter node.

        Returns:
            None
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        result = self.model.predict(tmp_path, verbose=False)[0]
        os.unlink(tmp_path)

        top1_idx = result.probs.top1
        top1_name = result.names[top1_idx]
        top1_conf = float(result.probs.top1conf)

        top5_idx = result.probs.top5
        top5 = [
            {"species": result.names[i], "confidence": round(float(result.probs.data[i]), 4)}
            for i in top5_idx
        ]

        payload = json.dumps({
            "species": top1_name,
            "confidence": round(top1_conf, 4),
            "top5": top5
        })

        species_msg = String()
        species_msg.data = payload
        self.species_publisher.publish(species_msg)

        self.get_logger().info(f'Species: {top1_name} (conf {top1_conf:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = SpeciesClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
