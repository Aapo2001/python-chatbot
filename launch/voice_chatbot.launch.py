from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    config_path = LaunchConfiguration("config_path")
    load_config_file = LaunchConfiguration("load_config_file")
    enable_voice_loop = LaunchConfiguration("enable_voice_loop")
    enable_tts_playback = LaunchConfiguration("enable_tts_playback")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value="config.json"),
            DeclareLaunchArgument("load_config_file", default_value="true"),
            DeclareLaunchArgument("enable_voice_loop", default_value="false"),
            DeclareLaunchArgument("enable_tts_playback", default_value="true"),
            Node(
                package="voice_chatbot_ros",
                executable="voice_chatbot_node",
                name="voice_chatbot",
                output="screen",
                parameters=[
                    {
                        "config_path": ParameterValue(config_path, value_type=str),
                        "load_config_file": ParameterValue(
                            load_config_file, value_type=bool
                        ),
                        "enable_voice_loop": ParameterValue(
                            enable_voice_loop, value_type=bool
                        ),
                        "enable_tts_playback": ParameterValue(
                            enable_tts_playback, value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
