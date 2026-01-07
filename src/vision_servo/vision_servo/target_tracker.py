import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
# [新增] 引入 PositionTarget 消息，用于混合控制
from mavros_msgs.msg import State, PositionTarget 

import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import math  # [新增] 用于坐标转换
from ultralytics import YOLO
import time
import threading

class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')
        
        # =========================
        # 1. 参数设置
        # =========================
        self.model_path = '/home/orinnx/last.pt' 
        
        self.target_dist = 1.0      # 视觉保持距离 (米)
        self.target_height = 1.0    # [修改] 目标高度 (直接作为位置设定点)
        self.conf_threshold = 0.5 
        
        # PID 参数 (移除 kp_height，因为高度由飞控接管)
        self.kp_dist = 0.4          # 前后速度 P增益
        self.kp_yaw = 0.5           # 偏航角速度 P增益
        
        # 速度限制 (水平)
        self.max_vx = 0.4           # 最大水平速度 m/s
        self.max_yaw = 0.4          # 最大偏航角速度 rad/s
        # [注意] 垂直速度限制 max_vz 已移除，请在 QGC 的 MPC_Z_VEL_MAX_UP 中设置
  
        # 状态标志位
        self.enable_tracking = False 
        self.current_z_height = 0.0
        self.current_yaw = 0.0      # [新增] 记录当前航向，用于坐标转换
        self.current_state = State() 
        self.intrinsics = None      

        # 初始化视觉计算出的速度指令
        self.vision_vx = 0.0
        self.vision_wz = 0.0

        self.bridge = CvBridge()
        self.service_cb_group = ReentrantCallbackGroup()

        # =========================
        # 2. 模型加载
        # =========================
        self.get_logger().info(f'正在加载 YOLO 模型: {self.model_path} ...')
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info('YOLO 模型加载成功!')
        except Exception as e:
            self.get_logger().error(f'模型加载失败! {e}')
            raise e

        # =========================
        # 3. QoS 配置
        # =========================
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # =========================
        # 4. 订阅与发布
        # =========================
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_best_effort)
        self.create_subscription(State, '/mavros/state', self.state_callback, qos_best_effort)
        self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        
        # [修改] 改为发布 setpoint_raw，不再发布 setpoint_velocity
        self.target_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=self.service_cb_group)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode', callback_group=self.service_cb_group)

        # 图像同步
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        # 定时器
        self.cmd_timer = self.create_timer(0.05, self.cmd_timer_callback)
        self.log_timer = self.create_timer(2.0, self.log_timer_callback)

        self.get_logger().info('混合控制节点已启动: 高度(位置) + 水平(速度)')

    # --------- 回调函数 ---------
    def state_callback(self, msg):
        self.current_state = msg

    def local_pose_callback(self, msg):
        self.current_z_height = msg.pose.position.z
        
        # [关键新增] 提取四元数并转为偏航角 (Yaw)
        # 视觉计算出的是“向前飞”，我们需要根据当前机头朝向，算出是向东还是向北飞
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
        height = self.current_z_height
        track_status = "开启 (视觉追踪中)" if self.enable_tracking else "关闭 (仅定高)"
        
        vision_info = ""
        if self.enable_tracking:
            vision_info = f" | 视觉指令 [Vx:{self.vision_vx:.2f}, Wz:{self.vision_wz:.2f}]"

        print(f"[{time.strftime('%H:%M:%S')}] 模式: {mode} | 高度: {height:.2f}m (目标: {self.target_height}m){vision_info} | 追踪: {track_status}")
        
        if mode != "OFFBOARD" and self.enable_tracking:
             print("  [警告] 追踪已激活，但当前非 Offboard 模式！")

    # --------- 核心：混合控制 (20Hz) ---------
    def cmd_timer_callback(self):
        # 安全锁
        if self.current_state.mode != "OFFBOARD":
            self.vision_vx = 0.0
            self.vision_wz = 0.0
            return

        # 构建 PositionTarget 消息
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED # 使用局部坐标系 (东北天)

        # [核心] 设置 type_mask (掩码)
        # 1 = 忽略该维度， 0 = 启用该维度
        # 我们需要控制: Z位置 (PosZ), X速度 (Vx), Y速度 (Vy), 偏航角速度 (YawRate)
        # 我们要忽略: X位置, Y位置, Z速度, 所有加速度, 偏航角度(Yaw)
        # 掩码计算:
        # IGNORE_PX(1) + IGNORE_PY(2) + IGNORE_VZ(32) + 
        # IGNORE_AFX(64) + IGNORE_AFY(128) + IGNORE_AFZ(256) + IGNORE_YAW(1024) 
        # = 1 + 2 + 32 + 64 + 128 + 256 + 1024 = 1507
        msg.type_mask = 1507 

        # 1. 高度控制 (直接赋值位置)
        # 只要你在 QGC 把 MPC_Z_VEL_MAX_UP 设为 0.3，飞控就会以 0.3m/s 慢慢爬升到这个高度
        msg.position.z = self.target_height 

        # 2. 水平控制 (速度)
        if self.enable_tracking:
            # 坐标转换：将机体坐标系速度 (Body Frame) 转为 世界坐标系速度 (Local ENU)
            # body_vx 是 YOLO 算出来的“向前”速度
            body_vx = self.vision_vx
            
            # 旋转矩阵公式
            msg.velocity.x = body_vx * math.cos(self.current_yaw)
            msg.velocity.y = body_vx * math.sin(self.current_yaw)
            
            # 偏航角速度直接赋值
            msg.yaw_rate = self.vision_wz
        else:
            # 没开启追踪时，只定高，水平不动
            msg.velocity.x = 0.0
            msg.velocity.y = 0.0
            msg.yaw_rate = 0.0

        self.target_pub.publish(msg)

    # --------- 视觉处理 ---------
    def sync_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or not self.enable_tracking:
            self.vision_vx = 0.0
            self.vision_wz = 0.0
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            results = self.model(cv_image, verbose=False, conf=self.conf_threshold)

            temp_vx = 0.0
            temp_wz = 0.0

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                v = np.clip(v, 0, depth_image.shape[0] - 1)
                u = np.clip(u, 0, depth_image.shape[1] - 1)
                dist_mm = depth_image[v, u] 

                if dist_mm > 100 and dist_mm < 10000:
                    dist_m = dist_mm / 1000.0
                    Z_cam = dist_m
                    X_cam = (u - self.cx) * Z_cam / self.fx

                    # 水平方向依然使用 PID 控制速度
                    err_dist = Z_cam - self.target_dist
                    if abs(err_dist) > 0.1: 
                        temp_vx = np.clip(err_dist * self.kp_dist, -self.max_vx, self.max_vx)
                    
                    temp_wz = np.clip(-X_cam * self.kp_yaw, -self.max_yaw, self.max_yaw)
            
            self.vision_vx = temp_vx
            self.vision_wz = temp_wz

        except Exception as e:
            # 视觉处理偶尔出错不影响主循环
            pass

    # --------- 服务调用 ---------
    def arm_drone(self):
        if not self.arming_client.wait_for_service(timeout_sec=1.0): return False
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 2.0: return False
            time.sleep(0.01)
        return future.result().success

    def set_offboard_mode(self):
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0): return False
        req = SetMode.Request()
        req.custom_mode = "OFFBOARD"
        future = self.set_mode_client.call_async(req)
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 2.0: return False
            time.sleep(0.01)
        return future.result().mode_sent

def main(args=None):
    rclpy.init(args=args)
    
    print(">>> 正在初始化节点...")
    try:
        node = TargetTracker()
    except Exception as e:
        print(f">>> 初始化失败: {e}")
        return

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("---------------------------------------")
    print("混合控制模式就绪 (Z轴位置 + XY轴速度)")
    print("请确保已在 QGC 中设置 MPC_Z_VEL_MAX_UP = 0.3 m/s")
    print("---------------------------------------")
    time.sleep(1)

    # 1. 解锁
    print(">>> [1/4] 请求解锁...")
    if node.arm_drone():
        print(">>> 解锁成功!")
        time.sleep(0.5)
        
        # 2. 切 Offboard
        print(">>> [2/4] 请求切换 OFFBOARD...")
        if node.set_offboard_mode():
            print(">>> OFFBOARD 已激活，无人机将自动上升至 1.0m")
            
            # 3. 等待高度 (直接用位置判断)
            while rclpy.ok():
                if node.current_state.mode == "OFFBOARD":
                    # 因为是位置控制，我们不需要担心过冲，直接判断是否接近
                    if abs(node.current_z_height - node.target_height) < 0.15:
                        print("\n>>> [3/4] 已到达预定高度 <<<")
                        break
                time.sleep(0.2)

            print(">>> 悬停 3 秒...")
            time.sleep(1) # 这里改小了一点，3秒也可以
            
            print("\n>>> [4/4] 视觉追踪已激活！ <<<")
            node.enable_tracking = True
            
            try:
                while rclpy.ok(): time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print(">>> 切 Offboard 失败")
    else:
        print(">>> 解锁失败")

    # 退出发送停止指令
    stop_msg = PositionTarget()
    # 退出时掩码设为忽略所有，或者只发位置保持当前位置，这里直接销毁
    node.target_pub.publish(stop_msg)
    node.destroy_node()
    rclpy.shutdown()
    print(">>> 程序退出")

if __name__ == '__main__':
    main()