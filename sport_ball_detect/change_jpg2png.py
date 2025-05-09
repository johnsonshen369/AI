from PIL import Image
import os

# 定义转换函数
def convert_jpg_to_png_and_remove(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 只处理 .jpg 文件
        if filename.lower().endswith('.jpg'):
            # 构建完整的 .jpg 文件路径
            jpg_path = os.path.join(directory, filename)
            # 打开 .jpg 图像
            with Image.open(jpg_path) as img:
                # 构建 .png 文件名
                png_filename = os.path.splitext(filename)[0] + '.png'
                png_path = os.path.join(directory, png_filename)
                # 将图像转换为 .png 格式并保存
                img.save(png_path, 'PNG')
                print(f"Converted {filename} to {png_filename}")
            
            # 删除原 .jpg 文件
            os.remove(jpg_path)
            print(f"Deleted {filename}")

# 指定目录路径
directory_path = 'images/val'

# 调用转换函数
convert_jpg_to_png_and_remove(directory_path)