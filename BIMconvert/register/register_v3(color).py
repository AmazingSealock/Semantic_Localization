import open3d as o3d
import numpy as np

def load_point_cloud_with_labels(file_path):
    pcd = o3d.io.read_point_cloud(file_path, format='ply')
    print(f"Point cloud before transformation: {pcd}")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors

def create_open3d_point_cloud(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def perform_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.0

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )
    return result.transformation
    #return result.transformation, result.correspondence_set

def colored_icp(source, target, init_transformation, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50))
    return result

def visualize_point_clouds(source_pcd, target_pcd, transformed_source_pcd=None):
    source_pcd.paint_uniform_color([1, 0, 0])  # Red
    target_pcd.paint_uniform_color([0, 1, 0])  # Green
    pcds = [source_pcd, target_pcd]
    if transformed_source_pcd is not None:
        transformed_source_pcd.paint_uniform_color([0, 0, 1])  # Blue
        pcds.append(transformed_source_pcd)
    o3d.visualization.draw_geometries(pcds)

if __name__ == "__main__":
    source_file_path = "../cutply/transformed30.ply"
    target_file_path = "../ply/labeledW15F.ply"
    source_points, source_colors = load_point_cloud_with_labels(source_file_path)
    target_points, target_colors = load_point_cloud_with_labels(target_file_path)

    voxel_size = 0.5  # Example voxel size

    source_pcd = create_open3d_point_cloud(source_points, source_colors)
    target_pcd = create_open3d_point_cloud(target_points, target_colors)
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # 计算FPFH特征
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # 粗配准
    # print("Performing initial RANSAC registration...")
    init_transformation = perform_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # print("Initial RANSAC transformation:")
    # print(init_transformation)

    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 精配准
    # print("Performing Colored ICP registration...")
    result_icp = colored_icp(source_down, target_down, init_transformation, voxel_size)
    # print("Colored ICP refined transformation:")
    # print(result_icp.transformation)

    # Transform the source point cloud
    transformed_source_pcd = source_pcd.transform(result_icp.transformation)

    # Output the final transformation matrix
    # print("Final registration result:")
    print("Transformation matrix:\n", result_icp.transformation)

    # 打印配准后的RMSE和Fitness值
    print(f"ICP fitness: {result_icp.fitness}")
    print(f"ICP inlier RMSE: {result_icp.inlier_rmse}")

    dists = np.asarray(target_pcd.compute_point_cloud_distance(transformed_source_pcd))
    print(f"Mean distance after registration: {np.mean(dists)}")
    print(f"Standard deviation of distances: {np.std(dists)}")

    # Visualize the registered point clouds
    # print("Visualizing registered point clouds...")
    visualize_point_clouds(source_pcd, target_pcd, transformed_source_pcd)
