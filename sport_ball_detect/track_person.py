import cv2
import torch
import numpy as np
from deep_sort.deep_sort import DeepSort

# 初始化YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# DeepSORT初始化
deepsort = DeepSort('deep_sort/deep/checkpoint/ckpt.t7')

# 定义人形的标签
PERSON_CLASS = 'person'

# 增强对比度的函数
def enhance_contrast(frame):
    alpha = 1.5  # 对比度增强系数
    beta = 20    # 亮度增强值
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return enhanced_frame

# 检测人形并使用DeepSORT进行跟踪的函数
def detect_and_track_people(frame, deepsort):
    # 将帧从 BGR 转换为 RGB 格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 对图像进行推理
    results = model(img)
    
    # 获取检测到的对象
    detections = results.pandas().xyxy[0]
    
    # 筛选出类别为 'person' 的对象
    persons = detections[detections['name'] == PERSON_CLASS]
    
    bbox_xywh = []
    confs = []

    # 遍历每个person并收集边界框和置信度
    for index, person in persons.iterrows():
        xmin, ymin, xmax, ymax = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        confidence = person['confidence']

        # 转换坐标为中心点格式 (x_center, y_center, width, height)
        bbox_xywh.append([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin])
        confs.append([confidence])

    # 使用DeepSORT进行跟踪
    outputs = deepsort.update(np.array(bbox_xywh), np.array(confs), frame)

    # 绘制跟踪框和ID
    for output in outputs:
        x1, y1, x2, y2, track_id = output[0], output[1], output[2], output[3], output[4]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# 加载视频
input_video = "clip2.mp4"
cap = cv2.VideoCapture(input_video)

# 获取视频帧率和帧尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

# 初始化视频写入对象
output_video = cv2.VideoWriter('output_with_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 处理并保存每一帧
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 增强对比度
    frame = enhance_contrast(frame)
    
    # 检测人形并进行跟踪
    frame_with_tracking = detect_and_track_people(frame, deepsort)
    
    # 将处理后的帧写入到输出视频
    output_video.write(frame_with_tracking)
    
    # 更新处理帧数
    frame_count += 1

    # 计算并打印进度
    progress = (frame_count / total_frames) * 100
    print(f"Processed frame {frame_count}/{total_frames} ({progress:.2f}%)")
    
    # 显示处理后的帧（可选）
    cv2.imshow('Person Detection and Tracking', frame_with_tracking)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和写入对象并关闭所有窗口
cap.release()
output_video.release()
cv2.destroyAllWindows()