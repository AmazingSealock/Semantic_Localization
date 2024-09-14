import re

# Step 1: Read the Bounding Box Data
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

# Step 2: Replace Names
def replace_names(bounding_boxes):
    for box in bounding_boxes:
        if 'Shape' in box['name'] or 'Line' in box['name']:
            box['name'] = 'floor'
        elif '门' in box['name'] or 'Object' in box['name']:
            box['name'] = 'door'
        elif '天花板' in box['name'] or '矩形梁' in box['name']:
            box['name'] = 'ceil'
        elif '墙' in box['name'] or '梃' in box['name'] or '板' in box['name'] or '梁' in box['name']:
            box['name'] = 'wall'
        elif '台' in box['name'] or '梯' in box['name'] or '钢' in box['name']:
            box['name'] = 'staircase'
        elif '栏' in box['name']:
            box['name'] = 'railing'
        elif '柱' in box['name'] or 'Rectangle' in box['name']:
            box['name'] = 'column'
        elif '窗' in box['name']:
            box['name'] = 'window'
        else :
            box['name'] = 'nothing'
    return bounding_boxes
    

# Step 3: Save the Updated Data
def save_bounding_boxes_to_txt(bounding_boxes, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for box in bounding_boxes:
            file.write(f"name: {box['name']}\n")
            file.write(f"min_bound: {box['min_bound']}\n")
            file.write(f"max_bound: {box['max_bound']}\n")

# Main Execution
bounding_boxes = read_bounding_boxes('./txt/bounding_boxes.txt')
bounding_boxes = replace_names(bounding_boxes)
save_bounding_boxes_to_txt(bounding_boxes, './txt/consolidated_bounding_boxes.txt')
