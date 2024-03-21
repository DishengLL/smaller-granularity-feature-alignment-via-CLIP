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
  
  def set_arg_parser(self):
    parser = argparse.ArgumentParser(description='parse input parameter for model configuration')
    parser.add_argument('--backbone', '-b', type=str,choices=["clip", "biomedclip","cxr-bert-s", "biovil-t"], help='the backbone module in the model')
    parser.add_argument('--prompt', type=str, help='the type of prompt used in the model training')
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
    
    
    
  
 