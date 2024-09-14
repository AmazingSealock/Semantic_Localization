import re
import open3d as o3d
import numpy as np

# Step 1: 读取Bounding Box信息并替换名称
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

# Step 2: 根据名称分配颜色
def get_color_for_label(label):
    colors = {
        'wall': [1, 0, 0],  # red
        'floor': [0, 1, 0],  # green
        'ceil': [0, 0, 1],  # blue
        'door': [1, 1, 0],  # yellow
        'railing': [1, 0, 1],  # magenta
        'column': [0, 1, 1],  # cyan
        'staircase': [0, 0, 0],  # black
        'window': [0.75, 0.25, 0.75],  # purple
        'nothing': [1, 1, 1],  # white
    }
    return colors.get(label, [1, 1, 1])  # white as default

# 根据标签名称获取标签编号
def get_label_number(label):
    labels = {
        'wall': 1,
        'floor': 2,
        'ceil': 3,
        'door': 4,
        'railing': 5,
        'column': 6,
        'staircase': 7,
        'window': 8,
        'nothing': 0,
    }
    return labels.get(label, 0)  # default label number is 0

# 标注点云
def label_point_cloud(point_cloud, bounding_boxes, expansion_factor=0.1):
    points = np.asarray(point_cloud.points)
    labels = np.zeros((points.shape[0], 3))  # Initialize labels with black color
    label_numbers = np.zeros((points.shape[0], 1))  # Initialize label numbers with 0

    # 设置默认标签和颜色
    default_label = 'floor'
    default_color = get_color_for_label(default_label)
    default_label_number = get_label_number(default_label)
    labels[:] = default_color
    label_numbers[:] = default_label_number

    for box in bounding_boxes:
        min_bound = np.array(box['min_bound'])
        max_bound = np.array(box['max_bound'])
        label_color = get_color_for_label(box['name'])
        label_number = get_label_number(box['name'])

        # 扩展包围盒
        expansion = (max_bound - min_bound) * expansion_factor #增加扩展因子
        min_bound = min_bound - expansion
        max_bound = max_bound + expansion
            
        # 获取包围盒内的点索引
        indices = np.where(
            (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
            (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
            (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
        )[0]

        # 仅更新包围盒内的点的颜色
        labels[indices] = label_color
        label_numbers[indices] = label_number
    
    point_cloud.colors = o3d.utility.Vector3dVector(labels)
    # point_cloud.labels = o3d.utility.Vector3dVector(label_numbers)
    return point_cloud, label_numbers


def create_bounding_box(min_bound, max_bound, color):
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    box.color = color
    return box


def visualize_bounding_boxes(bounding_boxes):
    geometries = []
    for box in bounding_boxes:
        color = get_color_for_label(box['name'])
        bbox = create_bounding_box(box['min_bound'], box['max_bound'], color)
        geometries.append(bbox)

    o3d.visualization.draw_geometries(geometries)

# 保存点云文件
def save_labeled_point_cloud(point_cloud, label_numbers, file_path, file_format='ply'):
    if file_format == 'ply':
        # Open3D does not support saving custom attributes directly to PLY.
        # We need to manually handle this.
        # Here, we use ASCII format for PLY with label numbers.
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
        labels = np.asarray(point_cloud.labels)
        data = np.hstack((points, colors, labels))
        np.savetxt(file_path, data, fmt='%.6f', delimiter=' ', encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# 主函数
def main():
    point_cloud_file = "./ply/W15F.ply"
    bounding_boxes_file = "./txt/consolidated_bounding_boxes.txt"
    output_file = "./ply/labeledW15F.ply"
    output_format = "ply"  # 可选值：'ply', 'pcd', 'xyz'

    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    print("read ply")
    print(point_cloud)

    # 读取Bounding Box数据
    bounding_boxes = read_bounding_boxes(bounding_boxes_file)
    print("read boxes")

    # 标注点云
    labeled_point_cloud, label_numbers = label_point_cloud(point_cloud, bounding_boxes)
    print("label points")

    # 保存标注过的点云文件
    save_labeled_point_cloud(labeled_point_cloud, label_numbers, output_file, output_format)
    print(f"Labeled point cloud saved as {output_format} format")

    # 可视化标注过的点云
    o3d.visualization.draw_geometries([labeled_point_cloud])

    # 可视化边界框
    visualize_bounding_boxes(bounding_boxes)

if __name__ == "__main__":
    main()