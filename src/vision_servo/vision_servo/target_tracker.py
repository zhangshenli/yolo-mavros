import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor # <--- 新增：多线程执行器
from rclpy.callback_groups import ReentrantCallbackGroup # <--- 新增：防堵车回调组
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
        
        # ==================== 配置区域 ====================
        self.model_path = '/home/orinnx/last.pt' 
        self.target_dist = 1.0      
        self.target_height = 1.0    
        
        self.kp_dist = 0.4          
        self.kp_yaw = 0.5           
        self.kp_height = 0.8        
        
        self.max_vx = 0.1          
        self.max_vz = 0.1           
        self.max_yaw = 0.1          
        # ================================================

        # --- 关键修改：创建一个“VIP通道” (可重入回调组) ---
        # 允许服务客户端的回调在其他回调运行时打断或并行处理
        self.service_cb_group = ReentrantCallbackGroup()

        self.get_logger().info(f'Loading YOLO model from: {self.model_path} ...')
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO Model Loaded Successfully!')

        self.bridge = CvBridge()
        self.intrinsics = None 
        self.current_z_height = 0.0

        # --- 订阅与发布 ---
        self.create_subscription(CameraInfo, '/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # --- 服务客户端 (绑定到 VIP 回调组) ---
        # 这样即使 YOLO 正在算图，这里也能收到“解锁成功”的消息
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming', callback_group=self.service_cb_group)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode', callback_group=self.service_cb_group)

        # --- 图像同步 ---
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        # --- 安全心跳 ---
        self.timer = self.create_timer(0.05, self.watchdog_callback)
        self.has_sent_first_cmd = False

        self.get_logger().info('Target Tracker Node Started.')

    def watchdog_callback(self):
        if not self.has_sent_first_cmd:
            self.vel_pub.publish(Twist())

    def local_pose_callback(self, msg):
        self.current_z_height = msg.pose.position.z

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = msg
            self.fx = msg.k[0]
            self.cx = msg.k[2]

    def sync_callback(self, color_msg, depth_msg):
        self.has_sent_first_cmd = True
        if self.intrinsics is None:
            return

        twist = Twist() 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # 这里的深度图处理如果不需要画图，可以简化以提速
            # depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") 
            
            results = self.model(cv_image, verbose=False)

            # A. 高度控制
            err_height = self.target_height - self.current_z_height
            twist.linear.z = np.clip(err_height * self.kp_height, -self.max_vz, self.max_vz)

            # B. 视觉控制
            if len(results[0].boxes) > 0:
                # 重新获取一次深度图(为了性能，只在检测到目标时转换)
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
                        twist.linear.x = np.clip(err_dist * self.kp_dist, -self.max_vx, self.max_vx)
                    
                    twist.angular.z = np.clip(-X_cam * self.kp_yaw, -self.max_yaw, self.max_yaw)
            
            self.vel_pub.publish(twist)
            # 简化打印，防止刷屏卡顿
            # print(f"\rCmd -> Vx:{twist.linear.x:.2f} | Vz:{twist.linear.z:.2f}", end="")

        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
            self.vel_pub.publish(Twist())

    def arm_drone(self):
        self.get_logger().info("正在呼叫解锁服务...")
        if not self.arming_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("解锁服务不在线！")
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
        self.get_logger().info("正在呼叫模式切换服务...")
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

    print("正在预热...")
    
    # 使用 MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2) 
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(2) 

    # === 修正部分开始 ===
    # 这里原来的 self.get_logger 全部改成了 node.get_logger
    if node.arm_drone():
        node.get_logger().info(">>> 解锁成功！<<<") 
        time.sleep(0.1) 
        
        if node.set_offboard_mode():
             node.get_logger().info(">>> 起飞！(OFFBOARD) <<<")
        else:
             node.get_logger().error("切模式失败，请手动切！")
    else:
        node.get_logger().error("解锁请求被拒绝！")
    # === 修正部分结束 ===

    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()