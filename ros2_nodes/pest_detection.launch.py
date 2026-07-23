"""
pest_detection.launch.py

ROS 2 launch file that starts the complete pest detection pipeline
with a single command. Launches all four nodes in the correct order:
  1. trap_detector_node  - finds yellow traps in camera frames
  2. binary_filter_node  - filters insect vs not-insect
  3. species_classifier_node - identifies the pest species
  4. heatmap_node        - generates field heatmap output
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='pest_detection_nodes',
            executable='trap_detector_node',
            name='trap_detector',
            output='screen',
            parameters=[]
        ),

        Node(
            package='pest_detection_nodes',
            executable='binary_filter_node',
            name='binary_filter',
            output='screen',
            parameters=[{
                'model_path': '/home/nikit/farm-ng-pest-model/models/checkpoints/binary_insect_classifier/weights/best.pt'
            }]
        ),

        Node(
            package='pest_detection_nodes',
            executable='species_classifier_node',
            name='species_classifier',
            output='screen',
            parameters=[{
                'model_path': '/home/nikit/farm-ng-pest-model/models/checkpoints/best.pt'
            }]
        ),

        Node(
            package='pest_detection_nodes',
            executable='heatmap_node',
            name='heatmap',
            output='screen',
            parameters=[{
                'output_dir': '/home/nikit/farm-ng-pest-model/results',
                'save_interval': 30
            }]
        ),

    ])
