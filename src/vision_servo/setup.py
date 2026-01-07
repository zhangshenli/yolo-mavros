from setuptools import setup

package_name = 'vision_servo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orinnx',
    maintainer_email='orinnx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    
    # === 重点修改这里 ===
    entry_points={
        'console_scripts': [
            
            'tracker = vision_servo.target_tracker:main',
            
            'takeoff = vision_servo.takeoff:main',
        ],
    },
)