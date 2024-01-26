import os 
import sys
import logging
from pathlib import Path

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
      
    
    
    
    
    
  
 