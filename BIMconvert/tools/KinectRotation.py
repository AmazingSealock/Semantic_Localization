import open3d as o3d
import numpy as np

# 读取原始的PLY文件
pcd = o3d.io.read_point_cloud("../ply/3.ply")

# 获取点云数据为Numpy数组
points = np.asarray(pcd.points)

# 只对XYZ坐标进行变换
# 变换矩阵定义为：将x=-x，y=-y
transformation_matrix = np.array([
    [1,  0, 0],
    [ 0, -1, 0],
    [ 0, 0, -1]
])

# 应用转换矩阵
converted_points = points @ transformation_matrix.T

# 更新点云数据
pcd.points = o3d.utility.Vector3dVector(converted_points)

# 读取原始的PLY文件内容
with open("../ply/3.ply", 'r') as file:
    ply_data = file.readlines()

# 找到header结束的位置
header_end_index = ply_data.index('end_header\n')

# 写入新的PLY文件
with open("../ply/output.ply", 'w') as file:
    # 写入header部分
    for line in ply_data[:header_end_index + 1]:
        file.write(line)
    
    # 写入点云数据部分
    for i in range(len(converted_points)):
        # 保留颜色和标签信息
        original_data = ply_data[header_end_index + 1 + i].split()
        x, y, z = converted_points[i]
        red, green, blue, label = original_data[3:7]
        file.write(f"{x} {y} {z} {red} {green} {blue} {label}\n")
