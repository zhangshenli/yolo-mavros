import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, PositionTarget 

import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from ultralytics import YOLO
import time
import threading

class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')
        
        # =========================
        # 1. 参数设置 (安全版)
        # =========================
        self.model_path = '/home/orinnx/last.pt' 
        self.target_dist = 1.0      # 保持 1.0米 距离
        self.target_height = 1.0    # 保持 1.0米 高度
        
        # [安全修改 1] PID 变得非常柔和
        self.kp_dist = 0.15         # 原 0.4 -> 0.15 (反应慢一点)
        self.kp_yaw = 0.2           # 原 0.5 -> 0.2 (转头慢一点)
        
        # [安全修改 2] 速度限制极低 (方便救机)
        self.max_vx = 0.15          # 最大速度 0.15 m/s
        self.max_yaw = 0.2          # 最大转速 0.2 rad/s
        
        # [安全修改 3] 忽略过远的目标 (防止误识别背景)
        self.ignore_dist_threshold = 4.0 

        # 状态变量
        self.enable_tracking = False 
        self.current_z_height = 0.0
        self.current_yaw = 0.0
        self.current_state = State() 
        self.intrinsics = None      

        # 视觉指令
        self.vision_vx = 0.0
        self.vision_wz = 0.0
        
        # 目标丢失计数器 (用于刹车)
        self.target_lost_counter = 0

        self.bridge = CvBridge()
        self.service_cb_group = ReentrantCallbackGroup()

        # =========================
        # 2. 加载 YOLO
        # =========================
        self.get_logger().info(f'正在加载 YOLO (安全模式): {self.model_path} ...')
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info('模型加载成功!')
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')

        # =========================
        # 3. MAVROS 通信
        # =========================
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_best_effort)
        self.create_subscription(State, '/mavros/state', self.state_callback, qos_best_effort)
        self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        
        self.target_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=self.service_cb_group)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode', callback_group=self.service_cb_group)

        # 图像处理
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        # 定时器
        self.cmd_timer = self.create_timer(0.05, self.cmd_timer_callback) # 20Hz
        self.log_timer = self.create_timer(1.0, self.log_timer_callback)  # 1Hz 日志

        self.get_logger().info('安全追踪节点已启动: 限速 0.15 m/s')

    # --------- 回调函数 ---------
    def state_callback(self, msg):
        self.current_state = msg

    def local_pose_callback(self, msg):
        self.current_z_height = msg.pose.position.z
        # 计算偏航角
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = msg
            self.fx = msg.k[0]
            self.cx = msg.k[2]

    def log_timer_callback(self):
        mode = self.current_state.mode
        h = self.current_z_height
        
        # 打印机头朝向 (0=东, 90=北)
        deg = math.degrees(self.current_yaw)
        
        status = "追踪中" if self.enable_tracking else "悬停"
        if self.target_lost_counter > 5: status = "目标丢失(刹车)"
        
        print(f"[{time.strftime('%H:%M:%S')}] {mode} | 高度:{h:.2f}m | 朝向:{deg:.0f}° | {status} | Vx:{self.vision_vx:.2f}")

    # --------- 控制循环 (20Hz) ---------
    def cmd_timer_callback(self):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED 
        msg.type_mask = 1507 

        # 1. 高度锁死 (位置控制)
        msg.position.z = self.target_height 

        # 2. 水平控制 (速度控制)
        # 必须同时满足: 开启开关 + OFFBOARD模式 + 目标未丢失
        if self.enable_tracking and \
           self.current_state.mode == "OFFBOARD" and \
           self.target_lost_counter < 10: # 如果连续0.3秒没看到人，就停
            
            # 坐标系旋转 (机体 -> 世界)
            #  - 帮助理解为什么要做这个计算
            # Vx_local = Vx_body * cos(yaw) - Vy_body * sin(yaw)
            body_vx = self.vision_vx
            
            msg.velocity.x = body_vx * math.cos(self.current_yaw)
            msg.velocity.y = body_vx * math.sin(self.current_yaw)
            msg.yaw_rate = self.vision_wz
        else:
            # 任何异常情况，全部刹车
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.yaw_rate = 0.0
            # 同时重置内部速度，防止下次接通时猛冲
            self.vision_vx = 0.0
            self.vision_wz = 0.0

        self.target_pub.publish(msg)

    # --------- 视觉处理 ---------
    def sync_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or not self.enable_tracking: return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            if not hasattr(self, 'model'): return
            
            results = self.model(cv_image, verbose=False, conf=0.5)

            # 默认：没检测到人 -> 速度归零，计数器+1
            target_found = False
            
            if len(results[0].boxes) > 0:
                # 只取置信度最高的一个
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # 边界保护
                v = np.clip(v, 0, depth_image.shape[0] - 1)
                u = np.clip(u, 0, depth_image.shape[1] - 1)
                
                dist_mm = depth_image[v, u] 
                
                # 距离有效性检查
                if dist_mm > 100 and dist_mm < (self.ignore_dist_threshold * 1000):
                    target_found = True
                    self.target_lost_counter = 0 # 重置计数器
                    
                    dist_m = dist_mm / 1000.0
                    Z_cam = dist_m
                    X_cam = (u - self.cx) * Z_cam / self.fx

                    # PID 计算
                    err_dist = Z_cam - self.target_dist
                    
                    # 死区设置 (0.15m内不动)
                    temp_vx = 0.0
                    if abs(err_dist) > 0.15: 
                        temp_vx = np.clip(err_dist * self.kp_dist, -self.max_vx, self.max_vx)
                    
                    temp_wz = np.clip(-X_cam * self.kp_yaw, -self.max_yaw, self.max_yaw)
                    
                    self.vision_vx = temp_vx
                    self.vision_wz = temp_wz

            if not target_found:
                # 如果这帧没看到人，计数器增加
                self.target_lost_counter += 1
                if self.target_lost_counter > 5:
                    self.vision_vx = 0.0
                    self.vision_wz = 0.0

        except Exception:
            pass

    # --------- 辅助函数 (保持不变) ---------
    def arm_drone(self):
        if not self.arming_client.wait_for_service(timeout_sec=1.0): return False
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > 2.0: return False
            time.sleep(0.01)
        return future.result().success

    def set_offboard_mode(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0): return False
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.set_mode_client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > 2.0: return False
            time.sleep(0.01)
        return future.result().mode_sent

def main(args=None):
    rclpy.init(args=args)
    node = TargetTracker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("---------------------------------------")
    print("【安全模式】系统启动")
    print("---------------------------------------")
    time.sleep(2)

    if node.arm_drone():
        print(">>> 解锁成功")
        time.sleep(0.5)
        if node.set_offboard_mode():
            print(">>> 起飞中...")
            while rclpy.ok():
                if node.current_state.mode == "OFFBOARD":
                    if abs(node.current_z_height - node.target_height) < 0.15:
                        break
                time.sleep(0.1)

            print(">>> 悬停 0.1 秒...")
            time.sleep(0.1)
            print(">>> 视觉追踪激活 (限速 0.15m/s)")
            node.enable_tracking = True
            
            try:
                while rclpy.ok(): time.sleep(1)
            except KeyboardInterrupt: pass

    stop_msg = PositionTarget()
    node.target_pub.publish(stop_msg)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()