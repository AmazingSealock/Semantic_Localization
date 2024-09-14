import numpy as np

def txt_to_ply(txt_file, ply_file):
    # 读取txt文件
    data = np.loadtxt(txt_file)
    
    # 提取xyz、rgb和label
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(int)
    
    # 将RGB值从0-1转换为0-255
    rgb = (rgb * 255).astype(np.uint8)
    
    # 获取顶点数
    num_vertices = len(data)
    
    # 写入PLY文件
    with open(ply_file, 'w') as ply:
        # 写入头部信息
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write(f"element vertex {num_vertices}\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        ply.write("property uchar red\n")
        ply.write("property uchar green\n")
        ply.write("property uchar blue\n")
        ply.write("property int label\n")
        ply.write("end_header\n")
        
        # 写入顶点信息
        for i in range(num_vertices):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            label = labels[i]
            ply.write(f"{x} {y} {z} {r} {g} {b} {label}\n")

# 使用示例
txt_file = '../txt/3.txt'
ply_file = '../ply/3.ply'
txt_to_ply(txt_file, ply_file)
