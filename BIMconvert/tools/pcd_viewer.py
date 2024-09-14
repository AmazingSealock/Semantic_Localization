import open3d as o3d

# 读取PCD文件
ply_file = "../ply/labeledW15F.ply"  # 替换为你的PLY文件路径
pcd = o3d.io.read_point_cloud(ply_file)
# 将点云的颜色设置为黑色
# pcd.paint_uniform_color([0, 0, 0])

# 打印点云信息
print(pcd)

# # 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

# ply_file = "../ply/fragment1.ply"  # 替换为你的PLY文件路径
# target_pcd = o3d.io.read_point_cloud(ply_file)
# ply_file = "../ply/labeledW15F.ply"  # 替换为你的PLY文件路径
# source_pcd = o3d.io.read_point_cloud(ply_file)

#  # 设置颜色和透明度
# target_pcd.paint_uniform_color([0, 1, 0])  # 目标点云涂成绿色
# source_pcd.paint_uniform_color([1, 0, 0])  # 源点云涂成红色

# # 可视化结果
# o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="配准结果", point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)


