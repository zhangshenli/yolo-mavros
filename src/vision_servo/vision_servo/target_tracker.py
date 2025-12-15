import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO  # <--- 新增：导入 YOLO

class TargetTracker(Node):
    def __init__(self):
        super().__init__('target_tracker')
        
        # ==========================================
        # TODO: 请在这里修改你的权重文件绝对路径
        # ==========================================
        self.model_path = '/home/orinnx/last.pt' 
        
        self.get_logger().info(f'Loading YOLO model from: {self.model_path} ...')
        # 加载模型，并强制使用 GPU (device=0)
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO Model Loaded Successfully!')

        # 初始化变量
        self.bridge = CvBridge()
        self.intrinsics = None 
        
        # 订阅 Camera Info
        self.create_subscription(
            CameraInfo,
            '/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10)

        # 创建订阅过滤器
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

        # 时间同步器
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info('Target Tracker Node Started. Waiting for camera info...')

    def camera_info_callback(self, msg):
        """ 获取相机内参 """
        if self.intrinsics is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsics = msg
            self.get_logger().info(f'Camera Intrinsics Loaded: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def sync_callback(self, color_msg, depth_msg):
        """ 核心回调函数 """
        if self.intrinsics is None:
            return

        try:
            # 1. 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 2. 执行 YOLOv8 推理
            # verbose=False 防止终端被 YOLO 的日志刷屏
            results = self.model(cv_image, verbose=False)

            # 检查是否检测到目标
            if len(results[0].boxes) > 0:
                # 策略：这里默认取置信度最高的第一个目标
                # 如果你想追踪特定类别，可以在这里加判断: if box.cls == 0: ...
                box = results[0].boxes[0]
                
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 计算中心点 (u, v)
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # 绘制边界框和中心点 (用于可视化)
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(cv_image, (u, v), 5, (0, 0, 255), -1)

                # 3. 读取深度距离
                # 注意 numpy 索引是 [行(y), 列(x)]
                dist_mm = depth_image[v, u] 

                # 过滤无效深度
                if dist_mm == 0:
                    # 简单的滤波：如果中心点深度为0，尝试搜索周围 5x5 区域的非零最小值或平均值
                    # 这里为了代码简单，暂时只打印警告
                    # self.get_logger().warn('Depth is 0 at center, skipping...')
                    pass
                else:
                    dist_m = dist_mm / 1000.0

                    # 4. 计算 3D 坐标 (相机坐标系)
                    Z_cam = dist_m
                    X_cam = (u - self.cx) * Z_cam / self.fx
                    Y_cam = (v - self.cy) * Z_cam / self.fy

                    self.get_logger().info(
                        f'Target Detected: XYZ_cam = [{X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}]'
                    )
                    
                    # 在图像上显示距离
                    text = f"X:{X_cam:.2f} Y:{Y_cam:.2f} Z:{Z_cam:.2f}m"
                    cv2.putText(cv_image, text, (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # ---------------------------------------------------
                    # 下一步预告：这里就是你发布给 MAVROS 的地方
                    # ---------------------------------------------------

            else:
                # 如果没检测到目标，可以在这里写逻辑 (比如悬停或旋转搜索)
                # self.get_logger().info('No target detected.')
                pass

            # 5. 显示结果
            cv2.imshow("YOLOv8 Tracking", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing images: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TargetTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()