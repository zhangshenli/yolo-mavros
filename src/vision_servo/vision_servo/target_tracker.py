import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading

class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')
        
        self.model_path = '/home/orinnx/last.pt' 
        self.target_dist = 1.0      
        self.target_height = 1.0    # 目标高度 1m
        
        # PID 参数
        self.kp_dist = 0.4          
        self.kp_yaw = 0.5           
        self.kp_height = 0.8        
        
        # 速度限制
        self.max_vx = 0.3           
        self.max_vz = 0.5           
        self.max_yaw = 0.3          
  

        # False: 只定高，水平不动 (起飞阶段)
        # True:  定高 + 视觉前后移动 (追踪阶段)
        self.enable_tracking = False 

        self.service_cb_group = ReentrantCallbackGroup()

        self.get_logger().info(f'Loading YOLO model from: {self.model_path} ...')
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO Model Loaded Successfully!')

        self.bridge = CvBridge()
        self.intrinsics = None 
        self.current_z_height = 0.0

        # 视觉计算出的临时速度变量
        self.vision_vx = 0.0
        self.vision_wz = 0.0

        # 订阅与发布
        self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # 服务客户端
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=self.service_cb_group)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode', callback_group=self.service_cb_group)

        # 图像处理
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        # 心跳定时器 (20Hz)
        # 注意：PX4要求进入Offboard前必须已有Setpoint数据流，所以这个定时器一启动就在发数据
        self.cmd_timer = self.create_timer(0.05, self.cmd_timer_callback)

        self.get_logger().info('Target Tracker Node Started.')

    def local_pose_callback(self, msg):
        self.current_z_height = msg.pose.position.z

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = msg
            self.fx = msg.k[0]
            self.cx = msg.k[2]

    # --- 线程1：运动控制核心 ---
    def cmd_timer_callback(self):
        twist = Twist()

        # 1. 高度控制 (始终运行，负责起飞和定高)
        err_height = self.target_height - self.current_z_height
        # 这里的 max_vz 限制了起飞速度
        twist.linear.z = np.clip(err_height * self.kp_height, -self.max_vz, self.max_vz)

        # 2. 水平控制 (受开关控制)
        if self.enable_tracking:
            # 开启追踪后，使用视觉计算的速度
            twist.linear.x = self.vision_vx
            twist.angular.z = self.vision_wz
        else:
            # 未开启追踪时 (起飞或悬停阶段)，水平速度置0
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0

        self.vel_pub.publish(twist)

    # --- 线程2：视觉处理 ---
    def sync_callback(self, color_msg, depth_msg):
        if self.intrinsics is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            results = self.model(cv_image, verbose=False)

            temp_vx = 0.0
            temp_wz = 0.0

            if len(results[0].boxes) > 0:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                dist_mm = depth_image[v, u] 
                if dist_mm > 0:
                    dist_m = dist_mm / 1000.0
                    Z_cam = dist_m
                    X_cam = (u - self.cx) * Z_cam / self.fx

                    err_dist = Z_cam - self.target_dist
                    if abs(err_dist) > 0.1: 
                        temp_vx = np.clip(err_dist * self.kp_dist, -self.max_vx, self.max_vx)
                    
                    temp_wz = np.clip(-X_cam * self.kp_yaw, -self.max_yaw, self.max_yaw)
            
            # 更新全局变量，供控制线程使用
            self.vision_vx = temp_vx
            self.vision_wz = temp_wz

        except Exception as e:
            self.get_logger().error(f'YOLO Error: {str(e)}')

    def arm_drone(self):
        self.get_logger().info("正在呼叫解锁服务...")
        if not self.arming_client.wait_for_service(timeout_sec=2.0):
            return False
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        while not future.done():
            time.sleep(0.01)
        try:
            return future.result().success
        except:
            return False

    def set_offboard_mode(self):
        self.get_logger().info("正在切换到 OFFBOARD 模式...")
        if not self.set_mode_client.wait_for_service(timeout_sec=2.0):
            return False
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.set_mode_client.call_async(req)
        while not future.done():
            time.sleep(0.01)
        try:
            return future.result().mode_sent
        except:
            return False

def main(args=None):
    rclpy.init(args=args)
    node = TargetTracker()

    print("---------------------------------------")
    print("Offboard Takeoff Logic")
    print("---------------------------------------")
    
    executor = MultiThreadedExecutor(num_threads=3) 
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 必须要等待一下，确保 cmd_timer_callback 已经运行了几次
    # 否则切换 OFFBOARD 会被拒绝 (Fail: no offboard signal found)
    time.sleep(1) 

    # === 第一步：解锁 (ARM) ===
    if node.arm_drone():
        node.get_logger().info(">>> 解锁成功 <<<") 
        time.sleep(0.2) 
        
        # === 第二步：直接切换 OFFBOARD 模式 (此时会自动起飞) ===
        # 原理：cmd_timer_callback 一直在算高度差，
        # 一旦切入 OFFBOARD，无人机就会立刻响应 Z轴速度，飞向 1m
        if node.set_offboard_mode():
            node.get_logger().info(">>> OFFBOARD 模式已激活，正在起飞至 1m... <<<")
            
            # === 第三步：等待到达指定高度 ===
            while True:
                current_h = node.current_z_height
                print(f"\r当前高度: {current_h:.2f}m / 1.0m", end="")
                
                # 当高度接近 0.95 米时认为到达
                if abs(current_h - node.target_height) < 0.1: 
                    print("\n>>> 已到达预定高度 <<<")
                    break
                time.sleep(0.1)

            # === 第四步：悬停 3 秒 ===
            print(">>> 悬停中 (3s) ... <<<")
            time.sleep(3.0)
            
            # === 第五步：激活 YOLO 追踪 ===
            print(">>> 激活视觉追踪！开始接近目标 <<<")
            node.enable_tracking = True
            
            # 保持主线程运行
            try:
                spin_thread.join()
            except KeyboardInterrupt:
                pass

        else:
             node.get_logger().error("切换 Offboard 模式失败！")
    else:
        node.get_logger().error("解锁失败")

    # 退出前的清理
    stop_msg = Twist()
    node.vel_pub.publish(stop_msg)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()