from glob import glob
from setuptools import setup

package_name = 'tello_vo'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        # numpy / matplotlib / opencv-python
    ],
    zip_safe=True,
    maintainer='vboxuser',
    maintainer_email='vboxuser@todo.todo',
    description='Monocular visual odometry node for Tello (ROS 2 Foxy)',
    license='Apache-2.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/rviz',   glob('rviz/*.rviz')),
    ],
    entry_points={
        'console_scripts': [
            'vo_node           = tello_vo.vo_node:main',
            'vo_node_control   = tello_vo.vo_node_control:main',
        ],
    },
)

