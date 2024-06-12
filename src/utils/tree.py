class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

def build_tree():
    
    no_finding = TreeNode("No Finding")
    heart_related = TreeNode("Heart Related Issues", 
                             [
                              TreeNode("Enlarged Cardiomediastinum"),
                              TreeNode("Cardiomegaly")
                              ]
                             )
    lung_issues = TreeNode("Lung Issues", 
                           [
                            TreeNode("Lung Lesion"),
                            TreeNode("Lung Opacity", 
                                     [
                                      TreeNode("Edema"),
                                      TreeNode("Consolidation"),
                                      TreeNode("Pneumonia")
                                      ]),
                              TreeNode("Atelectasis"),
                              TreeNode("Pneumothorax")
                            ])
    pleural_issues = TreeNode("Pleural Issues", [
                                                  TreeNode("Pleural Effusion"),
                                                  TreeNode("Pleural Other")
                                              ]
                              )
    other_issues = TreeNode("Other Issues", [
                                              TreeNode("Fracture"),
                                              TreeNode("Support Devices")
                                              ])
    root = TreeNode("Chest X-ray Findings", [no_finding, heart_related, lung_issues, pleural_issues, other_issues])
    return root

# 寻找最低公共祖先和计算节点到祖先的距离等函数保持不变


# 
# 寻找最低公共祖先
def find_lowest_common_ancestor(root, node1, node2):
    if root is None:
        return None

    if root.label == node1 or root.label == node2:
        return root

    common_ancestor = None
    for child in root.children:
        child_ancestor = find_lowest_common_ancestor(child, node1, node2)
        if child_ancestor:
            if common_ancestor:
                # 如果已经有一个公共祖先，说明当前节点为最低公共祖先
                return root
            else:
                # 否则更新为当前子树的公共祖先
                common_ancestor = child_ancestor

    return common_ancestor

# 计算两个节点之间的距离
def calculate_distance(root, node1, node2):
    # 寻找最低公共祖先
    lca = find_lowest_common_ancestor(root, node1, node2)

    # 计算两个节点到最低公共祖先的距离
    distance1 = find_distance_to_node(lca, node1, 0)
    distance2 = find_distance_to_node(lca, node2, 0)

    # 总距离为两者之和
    total_distance = distance1 + distance2
    return total_distance

# 计算节点到祖先的距离
def find_distance_to_node(current, target, distance):
    if current is None:
        return float('inf')

    if current.label == target:
        return distance

    for child in current.children:
        distance_to_child = find_distance_to_node(child, target, distance + 1)
        if distance_to_child != float('inf'):
            return distance_to_child

    return float('inf')

# 构建树
tree_root = build_tree()

# 选择两个节点
node1_label = "Enlarged Cardiomediastinum"
node2_label = "Fracture"

# 计算两个节点之间的距离
distance = calculate_distance(tree_root, node1_label, node2_label)

print(f"Distance between Node {node1_label} and Node {node2_label}: {distance}")


CHEXPERT_LABELS = [
 'Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Enlarged Cardiomediastinum',
 'Fracture',
 'Lung Lesion',
 'Lung Opacity',
 'No Finding',
 'Pleural Effusion',
 'Pleural Other',
 'Pneumonia',
 'Pneumothorax',
 "Support Devices"
]

import torch
import torch.nn.functional as F
n = len(CHEXPERT_LABELS)

import numpy as np

def create_negative_one_matrix(n):
    return np.full((n, n), -1)

negative_one_matrix = create_negative_one_matrix(n)
matrix = negative_one_matrix.copy()

# Print the created matrix
print(torch.tensor(negative_one_matrix))
for i, j in enumerate(CHEXPERT_LABELS):
  for x,y in enumerate(CHEXPERT_LABELS):
    node1_label = j
    node2_label = y
    distance = calculate_distance(tree_root, node1_label, node2_label)
    matrix[i][x] = distance
matrix = torch.tensor(matrix).float()


def safe_divide(a, b):
    # 使用 torch.where 处理除零情况
    result = torch.where(b != 0, a / b, torch.tensor(0.0))
    return result
  
# matrix = safe_divide(1, matrix)
normalized_matrix = F.normalize(matrix, p=2, dim=1)
# torch.save(normalized_matrix, './constants/normalized_distance_matrix.pt')