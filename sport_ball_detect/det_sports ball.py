import cv2
import torch
import numpy as np

# 使用本地训练的模型路径
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

# 定义篮球的标签（根据新模型的标签）
BASKETBALL_CLASS = 'sports ball'

# 增强对比度的函数
def enhance_contrast(frame):
    alpha = 1.5  # 对比度增强系数
    beta = 20    # 亮度增强值
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return enhanced_frame

# 检测篮球并绘制边界框的函数
def detect_and_draw_balls(frame):
    # 将帧从 BGR 转换为 RGB 格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 对图像进行推理
    results = model(img)
    
    # 获取检测到的对象
    detections = results.pandas().xyxy[0]
    
    # 筛选出类别为 'ball' 的对象
    basketballs = detections[detections['name'] == BASKETBALL_CLASS]
    
    # 遍历每个篮球并绘制边界框
    for index, basketball in basketballs.iterrows():
        xmin, ymin, xmax, ymax = int(basketball['xmin']), int(basketball['ymin']), int(basketball['xmax']), int(basketball['ymax'])
        confidence = basketball['confidence']

        print(f"Detected ball at: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}, confidence={confidence:.2f}")
        
        # 绘制矩形框
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 在边界框上方显示类别名称和置信度
        label = f"Ball {confidence:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

# 加载视频
input_video = "output_en.mp4"
cap = cv2.VideoCapture(input_video)

# 获取视频帧率和帧尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入对象
output_video = cv2.VideoWriter('output4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 处理并保存每一帧
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 增强对比度
    frame = enhance_contrast(frame)
    
    # 检测篮球并绘制边界框
    frame_with_boxes = detect_and_draw_balls(frame)
    
    # 将处理后的帧写入到输出视频
    output_video.write(frame_with_boxes)
    
    # 显示处理后的帧（可选）
    #cv2.imshow('Basketball Detection', frame_with_boxes)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和写入对象并关闭所有窗口
cap.release()
output_video.release()
cv2.destroyAllWindows()