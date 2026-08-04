"""Setup for the polyumi_pi_msgs package."""

import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'polyumi_pi_msgs'


def compile_protos():
    """
    Compile the protobuf files, generating .pyi type stubs when protoc supports them.

    The .pyi stubs need protoc's built-in --pyi_out generator, added in protobuf 3.20. We ask
    for it via grpcio-tools>=1.78 in this package's [build-system] requires, but that is only
    honoured by PEP 517 builds (pip install / python -m build). **colcon's ament_python build
    type runs this setup.py directly with the system interpreter**, so under `colcon build` the
    requirement is never installed and `grpc_tools` resolves to whatever the distro ships — on
    Ubuntu 24.04 that is grpcio-tools 1.14.1, bundling libprotoc 3.5.1. That protoc has no
    built-in pyi generator, so it looks for a `protoc-gen-pyi` plugin, finds none, and the whole
    build fails.

    The stubs are type hints only — nothing imports them at runtime — so a missing pyi generator
    should not break the build. Try with --pyi_out, and on failure fall back to generating just
    the _pb2.py modules with a warning. Version-sniffing protoc is avoided deliberately: its
    scheme has changed twice (3.20 -> 4.21 -> 25.x), whereas "try it and see" is version-proof.
    """
    package_dir = Path(__file__).resolve().parent
    proto_root = package_dir / package_name
    proto_files = sorted(proto_root.glob('*.proto'))

    base_cmd = [sys.executable, '-m', 'grpc_tools.protoc', f'-I={proto_root}']
    with_stubs = True
    for proto_file in proto_files:
        args = [f'--python_out={proto_root}', str(proto_file)]
        if with_stubs:
            result = subprocess.run(base_cmd + [f'--pyi_out={proto_root}'] + args)
            if result.returncode == 0:
                continue
            # Don't retry the probe for each remaining file — one failure settles it.
            with_stubs = False
            print(
                f'WARNING: {package_name}: protoc cannot generate .pyi stubs (needs protobuf '
                f'>= 3.20; this interpreter has an older grpc_tools). Falling back to _pb2.py '
                f'only — type hints will be stale or absent, but runtime is unaffected. '
                f'For stubs, build this package with pip/uv (which honours [build-system] '
                f'requires) or install a newer grpcio-tools for {sys.executable}.',
                file=sys.stderr,
            )
        subprocess.run(base_cmd + args, check=True)


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
