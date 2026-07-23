from setuptools import find_packages, setup

package_name = 'pest_detection_nodes'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pest_detection.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nikit',
    maintainer_email='sharm046@csusm.edu',
    description='Autonomous pest detection pipeline for farm-ng Amiga',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trap_detector_node = pest_detection_nodes.trap_detector_node:main',
            'binary_filter_node = pest_detection_nodes.binary_filter_node:main',
            'species_classifier_node = pest_detection_nodes.species_classifier_node:main',
            'heatmap_node = pest_detection_nodes.heatmap_node:main',
        ],
    },
)
