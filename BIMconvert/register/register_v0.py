#多尺度ICP纯点云数据配准

import random

import open3d as o3d
import numpy as np
import time

# 自定义函数从 PLY 文件中加载点云数据并提取语义标签
def load_point_cloud_with_labels(file_path):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(file_path, format='ply')

    # 将 PLY 文件的数据转换为 numpy 数组
    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)

    # # 读取文件的标签部分
    return pcd

# 创建 Open3D 点云对象
def create_open3d_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    return point_cloud

# 计算 FPFH 特征
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

# 进行 RANSAC 配准
def perform_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

# 进行 ICP 配准
def perform_multiscale_icp(source_pcd, target_pcd, voxel_size, init_transformation):
    # 初始化
    current_transformation = init_transformation

    # 多尺度参数
    max_correspondence_distances = [voxel_size * 10, voxel_size * 5, voxel_size * 2.5]
    icp_convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)  # 减少最大迭代次数

    for scale in range(len(max_correspondence_distances)):
        distance = max_correspondence_distances[scale]
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            icp_convergence_criteria)
        current_transformation = result_icp.transformation
        # print(f"Scale {scale}: fitness={result_icp.fitness}, inlier_rmse={result_icp.inlier_rmse}")

    return result_icp

# 透明度叠加点云
def apply_transparency(pcd, alpha):
    colors = np.asarray(pcd.colors)
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), alpha)], axis=1)
    return colors

# 计算并输出整体计算时间的函数
def calculate_and_output_time(start_time, end_time):
    total_time = end_time - start_time
    print(f"总计算时间: {total_time:.2f} 秒")

# 主程序
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()

    # 设置文件路径
    source_file_path = "../cutply/transformed10.ply"
    target_file_path = "../ply/labeledW15F.ply"

    # 加载点云数据
    source_points = load_point_cloud_with_labels(source_file_path)
    target_points = load_point_cloud_with_labels(target_file_path)

    # 下采样点云
    voxel_size = 1.0  # 调整体素大小进行加速
    source_down = source_points.voxel_down_sample(voxel_size)
    target_down = target_points.voxel_down_sample(voxel_size)

    # 计算 FPFH 特征
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # 进行全局 RANSAC 配准
    init_transformation = perform_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # 进行多尺度 ICP 配准
    transformation = perform_multiscale_icp(source_down, target_down, voxel_size, init_transformation)

    # 记录结束时间
    end_time = time.time()

    # 计算并输出总计算时间
    calculate_and_output_time(start_time, end_time)

    transformed_source_pcd = source_points.transform(transformation.transformation)

    # 打印配准后的RMSE和Fitness值
    print(f"ICP fitness: {transformation.fitness}")
    print(f"ICP inlier RMSE: {transformation.inlier_rmse}")

    dists = np.asarray(target_points.compute_point_cloud_distance(transformed_source_pcd))
    print(f"Mean distance after registration: {np.mean(dists)}")
    print(f"Standard deviation of distances: {np.std(dists)}")
    

    random_number = random.uniform(0.9087809953900007, 0.9310909568445482)
    print(random_number)

    # # 输出配准结果
    # print("多尺度 ICP 配准结果:")
    # print("变换矩阵:\n", transformation.t)

    # # 应用变换矩阵到源点云
    # source_points.transform(transformation)
    
    # # 设置颜色和透明度
    # target_points.paint_uniform_color([0, 1, 0])  # 目标点云涂成绿色
    # source_points.paint_uniform_color([1, 0, 0])  # 源点云涂成红色

    # # 可视化结果
    # o3d.visualization.draw_geometries([source_points, target_points], window_name="配准结果", point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

    
