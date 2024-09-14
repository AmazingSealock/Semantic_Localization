import open3d as o3d
import numpy as np

# 定义标签和对应的颜色
label_colors = {
    1: [255, 0, 0],       # wall: red
    2: [0, 255, 0],       # floor: green
    3: [0, 0, 255],       # ceil: blue
    4: [255, 255, 0],     # door: yellow
    5: [255, 0, 255],     # railing: magenta
    6: [0, 255, 255],     # column: cyan
    7: [0, 0, 0],         # staircase: black
    8: [191, 64, 191],    # window: purple
    0: [255, 255, 255]    # nothing: white
}

def modify_ply_colors(input_file, output_file):
    # 读取点云数据
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)

    # 读取标签
    with open(input_file, 'r') as f:
        lines = f.readlines()
        # 自动检测PLY文件的头部行数
        ply_header_lines = 0
        for line in lines:
            ply_header_lines += 1
            if line.strip() == "end_header":
                break

        labels = []
        for line in lines[ply_header_lines:]:
            labels.append(int(line.strip().split()[-1]))

    labels = np.array(labels)

    # 创建颜色数组
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)

    # 根据标签设置颜色
    for label, color in label_colors.items():
        mask = (labels == label)
        colors[mask] = color

    # 修改点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 缩放到0-1之间

    # 重新构建包含标签的点云数据，并保存为ASCII格式
    with open(output_file, 'w') as f:
        # 写入PLY头部
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property int label\n')
        f.write('end_header\n')
        
        # 写入点云数据和颜色及标签
        for i in range(len(points)):
            f.write(f'{points[i][0]} {points[i][1]} {points[i][2]} {colors[i][0]} {colors[i][1]} {colors[i][2]} {labels[i]}\n')


# 文件路径
input_file = '../ply/output.ply'
output_file = '../ply/colorPLY.ply'

# 执行修改颜色的函数
modify_ply_colors(input_file, output_file)
