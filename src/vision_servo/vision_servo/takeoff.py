import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
# 使用 PositionTarget 进行混合控制
from mavros_msgs.msg import State, PositionTarget 

import time
import threading

class SimpleHover(Node):
    def __init__(self):
        super().__init__('simple_hover')
        
        # =========================
        # 1. 参数设置
        # =========================
        self.target_height = 1.0    # 目标高度 1.0m
        
        # 状态变量
        self.current_z_height = 0.0
        self.current_state = State() 

        self.service_cb_group = ReentrantCallbackGroup()

        # =========================
        # 2. QoS 配置 (必须匹配 MAVROS)
        # =========================
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # =========================
        # 3. 订阅与发布
        # =========================
        # 订阅高度
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, qos_best_effort)
        # 订阅状态
        self.create_subscription(State, '/mavros/state', self.state_callback, qos_best_effort)
        
        # 发布控制指令 (setpoint_raw)
        self.target_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        # 服务客户端
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=self.service_cb_group)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode', callback_group=self.service_cb_group)

        # 定时器 (20Hz 发送指令)
        self.cmd_timer = self.create_timer(0.05, self.cmd_timer_callback)
        # 定时器 (1Hz 打印日志)
        self.log_timer = self.create_timer(1.0, self.log_timer_callback)

        self.get_logger().info('起飞悬停节点已启动 (去除视觉版)')

    # --------- 回调函数 ---------
    def state_callback(self, msg):
        self.current_state = msg

    def local_pose_callback(self, msg):
        self.current_z_height = msg.pose.position.z

    def log_timer_callback(self):
        mode = self.current_state.mode
        height = self.current_z_height
        print(f"[{time.strftime('%H:%M:%S')}] 模式: {mode} | 高度: {height:.2f}m (目标: {self.target_height}m)")
        
        if mode != "OFFBOARD":
             print("  Waiting for OFFBOARD...")

    # --------- 核心控制逻辑 (20Hz) ---------
    def cmd_timer_callback(self):
        # 安全锁：如果不小心切到了手动模式，这里停止发送指令(或者发空指令)，防止冲突
        if self.current_state.mode != "OFFBOARD":
            return

        # 构建混合控制消息
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED 

        # [掩码设置] 
        # 忽略: X位置, Y位置, Z速度, 加速度, 偏航角
        # 启用: Z位置(1.0m), X速度(0), Y速度(0), 偏航角速度(0)
        # Mask = 1507
        msg.type_mask = 1507 

        # 1. 高度控制 (位置)
        msg.position.z = self.target_height 

        # 2. 水平控制 (速度) -> 设为 0 让它悬停
        msg.velocity.x = 0.0
        msg.velocity.y = 0.0
        msg.yaw_rate = 0.0

        self.target_pub.publish(msg)

    # --------- 服务调用 ---------
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
    node = SimpleHover()
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("---------------------------------------")
    print("准备起飞至 1.0m 并悬停")
    print("---------------------------------------")
    time.sleep(1)

    # 1. 解锁
    print(">>> 请求解锁...")
    if node.arm_drone():
        print(">>> 解锁成功")
        time.sleep(0.5)
        
        # 2. 切 Offboard
        print(">>> 请求切换 OFFBOARD...")
        if node.set_offboard_mode():
            print(">>> OFFBOARD 激活! 无人机开始上升...")
            
            # 3. 只是死循环保持程序运行，打印状态
            try:
                while rclpy.ok():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print(">>> 切 Offboard 失败")
    else:
        print(">>> 解锁失败")

    # 退出
    stop_msg = PositionTarget()
    node.target_pub.publish(stop_msg)
    node.destroy_node()
    rclpy.shutdown()
    print(">>> 程序退出")

if __name__ == '__main__':
    main()