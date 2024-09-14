import open3d as o3d
import numpy as np
import re

# 读取Bounding Box信息并替换名称
def read_bounding_boxes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        
    pattern = re.compile(r'name: (.+?)\nmin_bound: \[(.+?)\]\nmax_bound: \[(.+?)\]', re.DOTALL)
    matches = pattern.findall(data)
    
    bounding_boxes = []
    for match in matches:
        name, min_bound, max_bound = match
        min_bound = list(map(float, min_bound.split(',')))
        max_bound = list(map(float, max_bound.split(',')))
        bounding_boxes.append({'name': name, 'min_bound': min_bound, 'max_bound': max_bound})
    
    return bounding_boxes

# 根据标签名称获取颜色
def get_color_for_label(label):
    colors = {
        'wall': [1, 0, 0],  # red
        'door': [0, 1, 0],  # green
        'board': [0, 0, 1],  # blue
        'staircase': [1, 1, 0],  # yellow
        'beam': [1, 0, 1],  # magenta
        'bar': [0, 1, 1],  # cyan
        'column': [0.5, 0.5, 0.5],  # grey
        'platform': [0.5, 0.25, 0],  # brown
        'window': [0, 0.5, 0.5],  # teal
        'ting': [0.75, 0.75, 0.75],  # light grey
        'shape': [0.75, 0, 0.25],  # purple
        'line': [0, 0.75, 0.25],  # greenish
        'object': [0.75, 0.25, 0.75],  # pink
        'point': [0.25, 0.25, 0.75],  # bluish
        'rectangle': [0.5, 0.75, 0.5],  # pale green
        'curtain': [0.75, 0.5, 0.5],  # light red
        'iron': [0.0, 0.0, 0.0],  # black
    }
    return colors.get(label, [1, 1, 1])  # white as default

# 根据标签名称获取标签编号
def get_label_number(label):
    labels = {
        'wall': 1,
        'door': 2,
        'board': 3,
        'staircase': 4,
        'beam': 5,
        'bar': 6,
        'column': 7,
        'platform': 8,
        'window': 9,
        'ting': 10,
        'shape': 11,
        'line': 12,
        'object': 13,
        'point': 14,
        'rectangle': 15,
        'curtain': 16,
        'iron': 17,
    }
    return labels.get(label, 0)  # default label number is 0

# 标注点云
def label_point_cloud(point_cloud, bounding_boxes):
    points = np.asarray(point_cloud.points)
    labels = np.zeros((points.shape[0], 3))  # Initialize labels with black color
    label_numbers = np.zeros((points.shape[0], 1))  # Initialize label numbers with 0

    for box in bounding_boxes:
        min_bound = box['min_bound']
        max_bound = box['max_bound']
        label_color = get_color_for_label(box['name'])
        label_number = get_label_number(box['name'])
        
        # 获取包围盒内的点索引
        indices = np.where(
            (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
            (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
            (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
        )[0]

        # 仅更新包围盒内的点的颜色和标签编号
        labels[indices] = label_color
        label_numbers[indices] = label_number
    
    point_cloud.colors = o3d.utility.Vector3dVector(labels)
    return point_cloud, label_numbers

# 保存带颜色和标签的点云文件
def save_labeled_point_cloud(point_cloud, label_numbers, file_path, file_format='ply'):
    if file_format == 'ply':
        # 使用ASCII格式保存带标签编号的PLY文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud.points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property int label\n")
            f.write("end_header\n")
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors) * 255
            labels = label_numbers
            for p, c, l in zip(points, colors, labels):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} {int(l[0])}\n")
    elif file_format == 'pcd':
        o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=True)
    elif file_format == 'xyz':
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        labels = label_numbers
        data = np.hstack((points, colors, labels))
        np.savetxt(file_path, data, fmt='%.6f', delimiter=' ', encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# 保存仅包含XYZ和标签的点云文件
def save_xyz_label_point_cloud(points, labels, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for p, l in zip(points, labels):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(l[0])}\n")

# 主函数
def main():
    point_cloud_file = ".\scans.ply"
    bounding_boxes_file = "consolidated_bounding_boxes.txt"
    output_file = "labeledXYZRGBIscans.ply"
    xyz_label_output_file = "labeledXYZscans.ply"
    output_format = "ply"  # 可选值：'ply', 'pcd', 'xyz'

    # 读取点云文件
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    print("Point cloud read")

    # 读取Bounding Box数据
    bounding_boxes = read_bounding_boxes(bounding_boxes_file)

    # 标注点云
    labeled_point_cloud, label_numbers = label_point_cloud(point_cloud, bounding_boxes)
    print("Point cloud labeled")

    # 保存带颜色和标签的点云文件
    save_labeled_point_cloud(labeled_point_cloud, label_numbers, output_file, output_format)
    print(f"Labeled point cloud saved as {output_format} format")

    # 保存仅包含XYZ和标签的点云文件
    save_xyz_label_point_cloud(np.asarray(labeled_point_cloud.points), label_numbers, xyz_label_output_file)
    print("XYZ and label point cloud saved as ply format")

if __name__ == "__main__":
    main()
