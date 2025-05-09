import os

def update_class_labels(directory, new_class_id):
    # 遍历目录中的所有子目录和文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # 修改类别
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        parts[0] = str(new_class_id)  # 修改类别ID为新的值
                        new_lines.append(' '.join(parts))
                
                # 将修改后的内容写回文件
                with open(file_path, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')

def main():
    base_directory = 'labels'  # 修改为你的标签目录路径
    new_class_id = 32
    subdirectories = ['train', 'val']
    
    for subdir in subdirectories:
        directory_path = os.path.join(base_directory, subdir)
        if os.path.exists(directory_path):
            update_class_labels(directory_path, new_class_id)
            print(f'Updated labels in {directory_path}')
        else:
            print(f'Directory {directory_path} does not exist')

if __name__ == "__main__":
    main()