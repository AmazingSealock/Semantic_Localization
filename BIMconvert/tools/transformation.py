import open3d as o3d
import numpy as np

# 读取带有颜色和标签的点云文件
def load_point_cloud_with_labels(file_path):
    pcd = o3d.io.read_point_cloud(file_path, format='ply')
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    labels = []
    with open(file_path, 'r') as file:
        header_ended = False
        while not header_ended:
            line = file.readline().strip()
            if line == "end_header":
                header_ended = True
        for line in file:
            parts = line.strip().split()
            if len(parts) == 7:  # XYZRGBL (标签在最后一列)
                labels.append(int(parts[-1]))
            elif len(parts) == 6:  # XYZRGB (没有标签)
                labels.append(0)  # 如果没有标签，赋值为0
            else:
                raise ValueError("点云文件的格式不符合预期")
    labels = np.array(labels)
    return points, colors, labels

# 读取只有XYZ信息的点云文件
def load_point_cloud_without_labels(file_path):
    pcd = o3d.io.read_point_cloud(file_path, format='ply')
    points = np.asarray(pcd.points)
    return points

def save_point_cloud(file_path, points):
    with open(file_path, 'w') as file:
        # 写入头部
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property int label\n")
        file.write("end_header\n")
        # 写入点云数据
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]} {int(point[3]*255)} {int(point[4]*255)} {int(point[5]*255)} {int(point[6])}\n")

# 合并点云数据将颜色和标签信息写入到只有XYZ信息的点云中
def merge_point_clouds(xyz_points, colors, labels):
    if len(xyz_points) != len(colors) or len(xyz_points) != len(labels):
        raise ValueError("点云文件中的点数量不一致")
    
    merged_points = np.hstack((xyz_points, colors, labels[:, np.newaxis]))
    
    return merged_points

if __name__ == "__main__":
    # 文件路径
    xyz_file_path = "scans2.ply"
    color_label_file_path = "labeledscans.ply"
    output_file_path = "merged.ply"
    
    # 加载点云数据
    xyz_points = load_point_cloud_without_labels(xyz_file_path)
    color_points, colors, labels = load_point_cloud_with_labels(color_label_file_path)
    
    # 确保点数量一致
    assert xyz_points.shape[0] == color_points.shape[0], "点云文件中的点数量不一致"
    
    # 合并点云数据
    merged_points = merge_point_clouds(xyz_points, colors, labels)
    
    # 保存合并后的点云
    save_point_cloud(output_file_path, merged_points)
    print(f"合并后的点云文件已保存到: {output_file_path}")
