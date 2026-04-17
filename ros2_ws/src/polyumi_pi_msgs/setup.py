"""Setup for the polyumi_pi_msgs package."""

import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'polyumi_pi_msgs'


def compile_protos():
    """Compile the protobuf files."""
    package_dir = Path(__file__).resolve().parent
    proto_root = package_dir / package_name
    proto_files = sorted(proto_root.glob('*.proto'))
    protoc = shutil.which('protoc')

    if protoc is None:
        raise RuntimeError(
            'protoc executable not found on PATH. Install protobuf-compiler.'
        )

    for proto_file in proto_files:
        proto_cmd = [
            protoc,
            f'-I={proto_root}',
            f'--pyi_out={proto_root}',
            f'--python_out={proto_root}',
            str(proto_file),
        ]
        subprocess.run(proto_cmd, check=True)


compile_protos()

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        package_name: ['*.proto', '*.pyi'],
    },
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
