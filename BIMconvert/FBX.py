      
import os, sys
import json
import open3d as o3d
import numpy as np
# 手动指定FBX SDK的路径
fbx_sdk_path = 'C:/Program Files/Autodesk/FBX/FBX SDK/2020.3.7/samples'

# 将FBX SDK的路径添加到sys.path中
sys.path.append(fbx_sdk_path)
from FbxCommon import FbxNodeAttribute
import FbxCommon
from fbx import *

def load_fbx(file_path):
    # Initialize the SDK manager and the scene
    sdk_manager, scene = FbxCommon.InitializeSdkObjects()
    # Load the scene from the specified file
    result = FbxCommon.LoadScene(sdk_manager, scene, file_path)
    
    if not result:
        raise Exception("Error loading scene")

    return scene

def extract_node_data(node):
    # Extract relevant information from the node
    node_data = {
        "name": node.GetName(),
        "bounding_box": None
    }

    # Check if the node has geometry (mesh)
    attribute = node.GetNodeAttribute()
    if attribute and attribute.GetAttributeType() == FbxNodeAttribute.EType.eMesh:
        mesh = attribute
        vertices = []
        # print(mesh)
        if mesh.GetControlPointsCount() == None:
            return node_data
        for i in range(mesh.GetControlPointsCount()):
            control_point = mesh.GetControlPointAt(i)
            vertices.append((control_point[0], control_point[1], control_point[2]))
        node_data["geometry"] = vertices

        vertices = np.array(vertices)
        if vertices.size > 0:
            min_bound = np.min(vertices, axis=0)
            max_bound = np.max(vertices, axis=0)
            node_data["bounding_box"] = (min_bound, max_bound)

    return node_data

def traverse_scene(scene):
    root_node = scene.GetRootNode()
    nodes_data = []

    def traverse(node):
        node_data = extract_node_data(node)
        if node_data["bounding_box"] is not None:
            nodes_data.append(node_data)
        for i in range(node.GetChildCount()):
            traverse(node.GetChild(i))

    traverse(root_node)
    return nodes_data

def save_bounding_boxes_to_txt(bounding_boxes, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for box in bounding_boxes:
            file.write(f"name: {box['name']}\n")
            file.write(f"min_bound: {box['bounding_box'][0].tolist()}\n")
            file.write(f"max_bound: {box['bounding_box'][1].tolist()}\n")


def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

def create_bounding_box(min_bound, max_bound):
    points = [
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
    ]
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    colors = [[0, 0, 0] for _ in lines]  # Red color for bounding box edges

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
    

# Path to your FBX file
fbx_file_path = './fbx/W15F.fbx'

# Load the FBX scene
scene = load_fbx(fbx_file_path)
print("read fbx")

# Traverse the scene and extract data
nodes_data = traverse_scene(scene)


txt_file_path = './txt/bounding_boxes.txt'
save_bounding_boxes_to_txt(nodes_data, txt_file_path)

bounding_boxes = [create_bounding_box(min_bound, max_bound) for node in nodes_data for min_bound, max_bound in [node["bounding_box"]]]
o3d.visualization.draw_geometries(bounding_boxes)




    