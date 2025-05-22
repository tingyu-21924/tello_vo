from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('tello_vo')
    rviz_cfg  = os.path.join(pkg_share, 'rviz', 'vo_path.rviz')

    return LaunchDescription([
        # (1) VO node
        Node(
            package='tello_vo',
            executable='vo_node_control',
            name='vo_node',
            output='screen',
        ),
        # (2) RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_cfg],
            output='screen',
        ),
    ])

