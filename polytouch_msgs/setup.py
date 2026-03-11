"""Setup for the polytouch_msgs package."""

from grpc_tools import protoc
from setuptools import find_packages, setup

package_name = 'polytouch_msgs'


def compile_protos():
    """Compile the protobuf files."""
    proto_file = 'polyumi_msgs/camera_frame.proto'
    protoc.main(
        [
            'grpc_tools.protoc',
            '-I=polyumi_msgs',
            '--python_out=polyumi_msgs',
            proto_file,
        ]
    )


compile_protos()

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='conorbot',
    maintainer_email='cwoodhayes@gmail.com',
    description='Protobuf messages for communication with PolyTouch CE Finger',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [],
    },
)
