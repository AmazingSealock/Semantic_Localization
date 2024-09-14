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
        'door': [0.5, 0.5, 0.5],  # grey
        'ceiling': [0, 1, 0],  # green
        'wall': [0, 0, 1],  # blue
        'staircase': [1, 1, 0],  # yellow
        'railing': [1, 0, 1],  # magenta
        'column': [0, 1, 1],  # cyan
        'floor': [1, 0, 0],  # red
        'window': [0.75, 0.25, 0.75],  # pink
        'nothing': [1, 1, 1],  # white
    }
    return colors.get(label, [1, 1, 1])  # white as default

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

# Main Execution
bounding_boxes = read_bounding_boxes('./txt/consolidated_bounding_boxes.txt')
visualize_bounding_boxes(bounding_boxes)
