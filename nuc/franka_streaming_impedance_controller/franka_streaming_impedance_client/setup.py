"""Setup for the franka_streaming_impedance_client package."""

from setuptools import find_packages, setup

package_name = 'franka_streaming_impedance_client'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='conorbot',
    maintainer_email='cwoodhayes@gmail.com',
    description='Producer-side chunk helpers for franka_streaming_impedance_controller.',
    license='MIT',
    tests_require=['pytest'],
)
