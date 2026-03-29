"""
``setup.py`` for the ``voice_chatbot_ros`` ROS 2 package.

Used **only** by ``colcon build`` (via ``ament_python``).  The standalone
``voice_chatbot`` pip package is defined in ``pyproject.toml`` and built
with ``python -m build``.

Entry points register ROS console scripts:

- ``voice_chatbot_app`` – standalone desktop GUI
- ``voice_chatbot_cli`` – terminal runner
- ``voice_chatbot_ros_app`` – ROS-connected desktop GUI
- ``voice_chatbot_setup_models`` – model bootstrap helper
- ``voice_chatbot_node`` – legacy monolithic node
- ``voice_stt_node`` – STT split node
- ``voice_llm_node`` – LLM split node
- ``voice_tts_node`` – TTS split node
"""

from glob import glob

from setuptools import find_packages, setup

package_name = "voice_chatbot_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(
        include=[
            "voice_chatbot",
            "voice_chatbot.*",
            package_name,
            f"{package_name}.*",
        ]
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    extras_require={
        "test": [
            "numpy",
            "pytest>=8,<10",
            "pytest-cov>=5,<8",
        ]
    },
    zip_safe=True,
    maintainer="Aapo",
    maintainer_email="noreply@example.com",
    description="ROS 2 Humble integration for the local voice chatbot pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "voice_chatbot_app = voice_chatbot.app:main",
            "voice_chatbot_cli = voice_chatbot.chatbot:main",
            "voice_chatbot_ros_app = voice_chatbot.ros_app:main",
            "voice_chatbot_setup_models = voice_chatbot.setup_models:main",
            "voice_chatbot_node = voice_chatbot_ros.node:main",
            "voice_stt_node = voice_chatbot_ros.stt_node:main",
            "voice_llm_node = voice_chatbot_ros.llm_node:main",
            "voice_tts_node = voice_chatbot_ros.tts_node:main",
        ],
    },
)
