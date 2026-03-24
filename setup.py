"""
``setup.py`` for the ``voice_chatbot_ros`` ROS 2 package.

Used by ``colcon build`` (via ``ament_python``) to install the ROS 2
nodes and their supporting Python modules into the local workspace
overlay.

Entry points register four console-script executables:

- ``voice_chatbot_node`` – legacy monolithic node
- ``voice_stt_node`` – STT split node
- ``voice_llm_node`` – LLM split node
- ``voice_tts_node`` – TTS split node

The ``py_modules`` list ensures that the shared library modules
(``audio_io``, ``config``, ``llm``, etc.) are also installed into the
overlay so the node scripts can import them.
"""

from glob import glob

from setuptools import find_packages, setup

package_name = "voice_chatbot_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    py_modules=[
        "audio_io",
        "config",
        "llm",
        "stt",
        "tts_engine",
        "vad",
    ],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="noreply@example.com",
    description="ROS 2 Humble integration for the local voice chatbot pipeline.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "voice_chatbot_node = voice_chatbot_ros.node:main",
            "voice_stt_node = voice_chatbot_ros.stt_node:main",
            "voice_llm_node = voice_chatbot_ros.llm_node:main",
            "voice_tts_node = voice_chatbot_ros.tts_node:main",
        ],
    },
)
