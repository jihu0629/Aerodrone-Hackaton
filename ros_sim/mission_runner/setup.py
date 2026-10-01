from setuptools import setup

package_name = 'mission_runner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['mission.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dohun',
    maintainer_email='ksw09129@gmail.com',
    description='맵핑+궤도 촬영 경로를 PX4 SITL에 OFFBOARD 세트포인트로 실행',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fly_mission = mission_runner.fly_mission:main',
        ],
    },
)
