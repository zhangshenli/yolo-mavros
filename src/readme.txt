/*启动相机节点 RGB+深度 */
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true \
    enable_depth:=true \
    enable_infra:=false \
    enable_infra1:=false \
    enable_infra2:=false \
    enable_gyro:=false \
    enable_accel:=false \
    align_depth.enable:=true

/*查看相机可视化*/
ros2 run rqt_image_view rqt_image_view

/*命令行启动conda环境以及指定python版本*/
load_drone

/*启动ros节点*/
ros2 run vision_servo tracker



/*重新编译*/
cd ~/drone_ws
rm -rf build install log
colcon build
source install/setup.bash


/*上传到github*/
cd ~/drone_ws/src
git add .
git commit -m "Updated code and fixed git submodule issue"
git push

/*启动Mavros节点*/
ros2 run mavros mavros_node --ros-args -p fcu_url:=serial:///dev/ttyACM0:57600 -p fcu_protocol:=v2.0