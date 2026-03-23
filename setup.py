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
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "voice_chatbot_node = voice_chatbot_ros.node:main",
        ],
    },
)
