"""Setup for the polyumi_pi_msgs package."""

import glob

from setuptools import find_packages, setup

package_name = 'polyumi_pi_msgs'


def compile_protos():
    """Compile the protobuf files."""
    from grpc_tools import protoc

    proto_files = glob.glob('polyumi_pi_msgs/*.proto')
    for proto_file in proto_files:
        protoc.main(
            [
                'grpc_tools.protoc',
                '-I=polyumi_pi_msgs',
                '--python_out=polyumi_pi_msgs',
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
