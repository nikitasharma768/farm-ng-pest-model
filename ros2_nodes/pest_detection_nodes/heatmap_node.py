"""
heatmap_node.py

ROS 2 node for the output stage of the pest detection pipeline.
Subscribes to species detection results and GPS coordinates, builds
a running detection log, and periodically saves both PNG and HTML
heatmap outputs.

Topics:
    Subscribes : /insect/species  (std_msgs/msg/String)
                 /gps/fix         (sensor_msgs/msg/NavSatFix)
    Publishes  : /heatmap/status  (std_msgs/msg/String)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class HeatmapNode(Node):
    """
    ROS 2 node that accumulates detection results and generates heatmap output.

    Purpose:
        Acts as the final stage in the ROS pipeline. Listens for species
        classification results and GPS fixes, builds a running detection
        log in memory, and periodically triggers heatmap generation.
        When GPS is unavailable (testing without Amiga), uses simulated
        coordinates.

    Subscribes:
        /insect/species (std_msgs/msg/String): JSON species result from
            the species classifier node.
        /gps/fix (sensor_msgs/msg/NavSatFix): GPS coordinates from the
            Amiga's RTK GPS module.

    Publishes:
        /heatmap/status (std_msgs/msg/String): Status message after each
            heatmap save.
    """

    def __init__(self):
        super().__init__('heatmap_node')

        self.declare_parameter('output_dir', '/home/nikit/farm-ng-pest-model/results')
        self.declare_parameter('save_interval', 10)
        self.output_dir = Path(self.get_parameter('output_dir').get_parameter_value().string_value)
        self.save_interval = self.get_parameter('save_interval').get_parameter_value().integer_value

        self.detection_log = []
        self.trap_counter = 0
        self.current_gps = None

        self.species_sub = self.create_subscription(
            String, '/insect/species', self.species_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, 10)

        self.status_publisher = self.create_publisher(String, '/heatmap/status', 10)

        self.timer = self.create_timer(
            float(self.save_interval), self.save_heatmap)

        self.get_logger().info(
            f'Heatmap node ready, saving every {self.save_interval}s to {self.output_dir}')

    def gps_callback(self, msg):
        """
        Store the latest GPS fix for tagging detection events.

        Args:
            msg (sensor_msgs/msg/NavSatFix): GPS message from Amiga.

        Returns:
            None
        """
        self.current_gps = {
            "latitude": msg.latitude,
            "longitude": msg.longitude
        }

    def species_callback(self, msg):
        """
        Record a new species detection event with GPS coordinates.

        Purpose:
            Each time a species result arrives, logs it with the current
            GPS position (or a simulated position if GPS is unavailable)
            into the running detection log. This log is what gets fed
            into heatmap.py to generate the output maps.

        Args:
            msg (std_msgs/msg/String): JSON string with species and
                confidence from the species classifier node.

        Returns:
            None
        """
        self.trap_counter += 1
        data = json.loads(msg.data)

        if self.current_gps:
            lat = self.current_gps["latitude"]
            lon = self.current_gps["longitude"]
        else:
            lat = 34.0522 + (self.trap_counter * 0.0003)
            lon = -117.2437 + (self.trap_counter * 0.0002)
            self.get_logger().warn('No GPS fix, using simulated coordinates')

        entry = {
            "trap_id": f"trap_{self.trap_counter:02d}",
            "latitude": lat,
            "longitude": lon,
            "trap_detected": True,
            "insect_count": 1,
            "detections": [{
                "species": data["species"],
                "confidence": data["confidence"]
            }]
        }
        self.detection_log.append(entry)
        self.get_logger().info(
            f'Logged detection: {data["species"]} at ({lat:.6f}, {lon:.6f})')

    def save_heatmap(self):
        """
        Periodically save the detection log and regenerate heatmaps.

        Purpose:
            Triggered every save_interval seconds. Writes the current
            detection log to JSON and calls heatmap.py and heatmap_html.py
            as subprocesses to regenerate both output formats. This keeps
            the heatmap up to date as the Amiga drives through the field.

        Args:
            None

        Returns:
            None
        """
        if not self.detection_log:
            self.get_logger().info('No detections yet, skipping heatmap save')
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / 'detection_log.json'

        full_log = {
            "run_timestamp": datetime.now().isoformat(),
            "total_images": self.trap_counter,
            "insects_found": len(self.detection_log),
            "results": self.detection_log
        }

        with open(log_path, 'w') as f:
            json.dump(full_log, f, indent=2)

        src_dir = Path('/home/nikit/farm-ng-pest-model/src')
        subprocess.run([sys.executable, str(src_dir / 'heatmap.py'),
                       '--input', str(log_path), '--output', str(self.output_dir)])
        subprocess.run([sys.executable, str(src_dir / 'heatmap_html.py'),
                       '--input', str(log_path), '--output', str(self.output_dir)])

        status_msg = String()
        status_msg.data = f'Heatmap saved: {len(self.detection_log)} detections'
        self.status_publisher.publish(status_msg)
        self.get_logger().info(status_msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = HeatmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
