import cv2
import torch
import os

# ���� YOLOv5 ģ��
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ��������ı�ǩ
BASKETBALL_CLASS = 'sports ball'

# �������ǰ 5 �룬����� 3 �����Ƶ����
pre_goal_duration = 5  # ����ǰ 5 ��
post_goal_duration = 3  # ����� 3 ��

# �������Ŀ¼
output_folder = "goal_clips"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ������������� (xmin, ymin, xmax, ymax)
# ��Ҫ����ʵ����Ƶ���ݵ���
basket_region = (654, 287, 748, 300)

# ������򲢻��Ʊ߽��ĺ���
def detect_and_draw_balls(frame):
    # ��ͼ��� BGR ת��Ϊ RGB ��ʽ
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ʹ�� YOLOv5 ģ�ͽ��м��
    results = model(img)
    
    # ��ȡ��⵽�Ķ���
    detections = results.pandas().xyxy[0]
    
    # ���˳����Ϊ 'sports ball' �Ķ���
    basketballs = detections[detections['name'] == BASKETBALL_CLASS]
    
    goal_detected = False  # ����Ƿ��⵽����

    # ����ÿ�����򲢻��Ʊ߿�
    for index, basketball in basketballs.iterrows():
        xmin, ymin, xmax, ymax = int(basketball['xmin']), int(basketball['ymin']), int(basketball['xmax']), int(basketball['ymax'])
        confidence = basketball['confidence']
        
        print(f"Detected ball at: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}, confidence={confidence:.2f}")
        
        # ������������ĵ�
        ball_center_x = (xmin + xmax) // 2
        ball_center_y = (ymin + ymax) // 2
        
        # ���ƾ��ο�
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # �ڱ߿��Ϸ���ʾ������ƺ����Ŷ�
        label = f"Ball {confidence:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # ��������Ƿ������������
        if (basket_region[0] <= ball_center_x <= basket_region[2] and
            basket_region[1] <= ball_center_y <= basket_region[3]):
            print("Ball entered the basket region!")
            goal_detected = True  # ��ǽ���״̬
    
    # ������������
    cv2.rectangle(frame, (basket_region[0], basket_region[1]), (basket_region[2], basket_region[3]), (255, 0, 0), 2)
    
    return frame, goal_detected  # ����ͼ����Ƿ����

# ������Ƶ
input_video = "clip2.mp4"
cap = cv2.VideoCapture(input_video)

# ��ȡ��Ƶ֡�ʺ�ͼ��ߴ�
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ��ʼ������
goal_detected_frames = []  # ������¼�����֡��
frame_number = 0

# ��������ÿһ֡
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # ������򲢻��Ʊ߽��ͬʱ����Ƿ����
    frame_with_boxes, goal_detected = detect_and_draw_balls(frame)
    
    # �����⵽���򣬽���ǰ֡�ű��浽�б���
    if goal_detected:
        goal_detected_frames.append(frame_number)
    
    # ��ʾ������֡����ѡ��
    # cv2.imshow('Basketball Detection', frame_with_boxes)
    
    # �� 'q' ���˳�
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

cap.release()

# �ͷ���ʾ����
cv2.destroyAllWindows()

# �����⵽������ȡ����ǰ 5 ��ͽ���� 3 �����ƵƬ��
if goal_detected_frames:
    cap = cv2.VideoCapture(input_video)
    
    for i, goal_frame in enumerate(goal_detected_frames):
        # ������ʼ֡�ͽ���֡
        start_frame = max(0, goal_frame - pre_goal_duration * fps)
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), goal_frame + post_goal_duration * fps)
        
        # ������Ƶ��ʼ֡
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # ��������ļ���
        output_file = os.path.join(output_folder, f"goal_clip_{i+1}.mp4")
        
        # ��ʼ����Ƶд�����
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # ��ȡ��д����ƵƬ��
        for frame_num in range(int(start_frame), int(end_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Saved clip {output_file}")
    
    cap.release()

print("All clips have been saved.")
