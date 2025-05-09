import cv2
import torch
import os

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 定义篮球的标签
BASKETBALL_CLASS = 'sports ball'

# 保存进球前 5 秒，进球后 3 秒的视频参数
pre_goal_duration = 5  # 进球前 5 秒
post_goal_duration = 3  # 进球后 3 秒

# 创建输出目录
output_folder = "goal_clips"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义篮筐的区域 (xmin, ymin, xmax, ymax)
# 需要根据实际视频内容调整
basket_region = (654, 287, 748, 300)

# 检测篮球并绘制边界框的函数
def detect_and_draw_balls(frame):
    # 将图像从 BGR 转换为 RGB 格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用 YOLOv5 模型进行检测
    results = model(img)
    
    # 获取检测到的对象
    detections = results.pandas().xyxy[0]
    
    # 过滤出类别为 'sports ball' 的对象
    basketballs = detections[detections['name'] == BASKETBALL_CLASS]
    
    goal_detected = False  # 标记是否检测到进球

    # 遍历每个篮球并绘制边框
    for index, basketball in basketballs.iterrows():
        xmin, ymin, xmax, ymax = int(basketball['xmin']), int(basketball['ymin']), int(basketball['xmax']), int(basketball['ymax'])
        confidence = basketball['confidence']
        
        print(f"Detected ball at: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}, confidence={confidence:.2f}")
        
        # 计算篮球的中心点
        ball_center_x = (xmin + xmax) // 2
        ball_center_y = (ymin + ymax) // 2
        
        # 绘制矩形框
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 在边框上方显示类别名称和置信度
        label = f"Ball {confidence:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # 检查篮球是否进入篮筐区域
        if (basket_region[0] <= ball_center_x <= basket_region[2] and
            basket_region[1] <= ball_center_y <= basket_region[3]):
            print("Ball entered the basket region!")
            goal_detected = True  # 标记进球状态
    
    # 绘制篮筐区域
    cv2.rectangle(frame, (basket_region[0], basket_region[1]), (basket_region[2], basket_region[3]), (255, 0, 0), 2)
    
    return frame, goal_detected  # 返回图像和是否进球

# 加载视频
input_video = "clip2.mp4"
cap = cv2.VideoCapture(input_video)

# 获取视频帧率和图像尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化变量
goal_detected_frames = []  # 用来记录进球的帧号
frame_number = 0

# 处理并保存每一帧
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 检测篮球并绘制边界框，同时检测是否进球
    frame_with_boxes, goal_detected = detect_and_draw_balls(frame)
    
    # 如果检测到进球，将当前帧号保存到列表中
    if goal_detected:
        goal_detected_frames.append(frame_number)
    
    # 显示处理后的帧（可选）
    # cv2.imshow('Basketball Detection', frame_with_boxes)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

cap.release()

# 释放显示窗口
cv2.destroyAllWindows()

# 如果检测到进球，提取进球前 5 秒和进球后 3 秒的视频片段
if goal_detected_frames:
    cap = cv2.VideoCapture(input_video)
    
    for i, goal_frame in enumerate(goal_detected_frames):
        # 计算起始帧和结束帧
        start_frame = max(0, goal_frame - pre_goal_duration * fps)
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), goal_frame + post_goal_duration * fps)
        
        # 设置视频起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 创建输出文件名
        output_file = os.path.join(output_folder, f"goal_clip_{i+1}.mp4")
        
        # 初始化视频写入对象
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # 提取并写入视频片段
        for frame_num in range(int(start_frame), int(end_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Saved clip {output_file}")
    
    cap.release()

print("All clips have been saved.")
