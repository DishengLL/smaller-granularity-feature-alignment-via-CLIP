import os 
import sys
import logging
from pathlib import Path
import argparse
import pdb
import random
import trace
import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import constants


logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为DEBUG，这里你可以根据需要设置不同的级别
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('utils.log')  # 输出到文件
    ]
)

class tools:
  def __init__(self):
    logging.info("initialize utils class")
    return 
  
  def mkdir(self, folder_path = None):
    if folder_path is None:
      raise ValueError("miss folder_path in mkdir_dir function!")
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
      logging.info(f"Folder '{folder_path}' created.")
      
class parser:
  def __init__(self):
    return
  "disease_level", "dis_diag_level", "dis_diag_des_level"
  def set_arg_parser(self):
    parser = argparse.ArgumentParser(description='parse input parameter for model configuration')
    parser.add_argument('--backbone', '-b', type=str,choices=["clip", "biomedclip","cxr-bert-s", "biovil-t"], help='the backbone module in the model')
    parser.add_argument('--prompt', type=str, choices=["dis_diag", "dis_diag_des", "basic"],default = "basic", help='the type of prompt used in the model training')
    parser.add_argument('--vision_only',"-vo", action='store_true', default=False, help='does the model contain vision branch')
    parser.add_argument('--backbone_v', "-bv", choices=['densenet'], type=str, help="vision encoder in image branch")
    parser.add_argument('--save_dir', type=str, help="the dir to save output")
    parser.add_argument('--learnable_weight',action='store_true', default=False, help='set learnable weights between differetn sub-losses(default: false)')
    parser.add_argument('--high_order',  "-ho", type=str,choices=["binary", "KL_based", "NA"], default="NA", help='using high-order correlation contrastive learning during training(default: false)')
    parser.add_argument('--two_phases',action='store_true', default=False, help='implement 2-phases training scheme') 
    parser.add_argument('--no_orthogonize',"-north", action='store_true', default=False, help='do not implement orthogonization operation in the whole pipeline')
    parser.add_argument('--no_contrastive',"-nc",  action='store_true', default=False, help='do not implement contrastive alignment between text and images')  
    parser.add_argument('--uncertain_based_weight', "-u", action='store_true', default=False, help='using uncertainty strategy to weight different sublosses(defualt: false)')  
    parser.add_argument('--weight_strategy', "-ws", type=str, choices=["uncertain_based_weight", "task_balance", "NA"], default="NA", help='choice different weighting strategies(default: NA)')  
    parser.add_argument('--labeling_strategy', "-LS", type=str,choices=["S1", "Original", "S2"], default="Original", help="specify the labeling strategy(default: Original - 3 labels)")
    parser.add_argument('--contrastive_param', "-CP", type=float, required=False, default= 1, help="specify the parameter for contrastive loss, which is the bridge between textual and visual branches.")
    parser.add_argument('--classification_param', "-ClsP", type=float, required=False, default= 1, help="specify the parameter for classification loss.")
    parser.add_argument('--orthogonal_param', "-OP", type=float, required=False, default= 1, help="specify the parameter for orthogonal loss.")
    parser.add_argument('--graph_param', "-GP", type=float, required=False, default= 1, help="specify the parameter for high-order loss.")
    parser.add_argument('--trainable_PLM', "-TP", type=int, required=False, default= 0, help="Specify the number of last few layers to be trainable.")
    parser.add_argument('--AP-PA-view', action='store_true', default = False, help="training and testing on AP and PA view position data")
    parser.add_argument('--trainable_VisionEncoder', action='store_true', default = False, help="all of vision encoder is trainable (initialize from large pretrained models)")
    parser.add_argument('--Alignment_Only', '-AO', action='store_true', default = False, help="Alignment visual and textual information only, this parameter is used to get the pretrained (evaluate the contrastive loss which could be reduced from 1.7 in the whole pipeline)")
    args = parser.parse_args() 
    return args
      
def set_random_seed(seed = 42):
  # set random seed 
  seed = seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  os.environ['PYTHONASHSEED'] = str(seed)
  
def set_env_config():
  torch.multiprocessing.set_start_method('spawn')# good solution !!!!
  logging.basicConfig(
  level=logging.DEBUG,  # 设置日志级别为DEBUG
  format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
  handlers=[
      logging.StreamHandler(),  # 输出到控制台
      logging.FileHandler('app.log')  # 输出到文件
      ]
  )
  logger = logging.getLogger('my_logger')
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  os.environ['TOKENIZERS_PARALLELISM']='false'
  return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 示例：计算模型中可训练参数的数量
# total_trainable_params = count_parameters(patch_embed)
# print("Total trainable parameters:", total_trainable_params)


def get_confusion_matrix(actual_labels:torch.Tensor, predicted_labels:torch.Tensor):
  if actual_labels.is_cuda: actual_labels = actual_labels.cpu()
  if predicted_labels.is_cuda: predicted_labels = predicted_labels.cpu() 
  # 计算多标签分类混淆矩阵
  mcm = multilabel_confusion_matrix(actual_labels, predicted_labels)
  return mcm

def get_Specificity_Precision_Recall_F1(actual_labels:torch.Tensor, predicted_labels:torch.Tensor):
  if actual_labels.is_cuda: actual_labels = actual_labels.cpu()
  if predicted_labels.is_cuda: predicted_labels = predicted_labels.cpu() 
  # specificity = []
  # for i in range(len(actual_labels[0])):
  #     tn = sum(1 for j in range(len(actual_labels)) if actual_labels[j][i] == 0 and predicted_labels[j][i] == 0)
  #     fp = sum(1 for j in range(len(actual_labels)) if actual_labels[j][i] == 0 and predicted_labels[j][i] == 1)
  #     specificity.append(tn / (tn + fp))

  # # 计算精确率
  # precision = precision_score(actual_labels, predicted_labels, average='macro')

  # # 计算召回率
  # recall = recall_score(actual_labels, predicted_labels, average='macro')

  # 计算F1分数
  f1 = f1_score(actual_labels, predicted_labels, average='samples')#, zero_division=1)

  return f1
  # return (specificity, precision, recall, f1)

import numpy as np
import matplotlib.pyplot as plt

# 疾病名称
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
def plot_confusion(mcm, suptitle = None):
  plt.ioff()

  confusion_matrices = mcm
  disease_names =  constants.CHEXPERT_LABELS

  # 创建子图
  num_plots = len(confusion_matrices)
  num_rows = int(np.ceil(num_plots / 3))  # 每行最多3个子图
  fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

  # 绘制每个混淆矩阵
  for i, (matrix, name) in enumerate(zip(confusion_matrices, disease_names)):
      row = i // 3
      col = i % 3
      ax = axes[row, col] if num_rows > 1 else axes[col]
      
      # 绘制混淆矩阵
      im = ax.imshow(matrix, cmap='Blues')
      
      # 添加标题和标签
      ax.set_title(name)
      ax.set_xticks(np.arange(2))
      ax.set_yticks(np.arange(2))
      ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
      ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
      
      # 在矩阵中显示数字
      for i in range(2):
          for j in range(2):
              text = ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

  # 添加颜色条
  cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
  fig.colorbar(im, cax=cbar_ax)

  # 调整子图间距
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
  if suptitle is not None: 
    plt.suptitle(suptitle)
  # 显示图像
  plt.show()


 