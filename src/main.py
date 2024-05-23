 import pdb, os
import random
import trace
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import LG_CLIP_LOSS
import constants
from dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, TestingCollator, TestingDataset
from models import MultiTaskModel
from train import Trainer
from evaluate import  Evaluator
import traceback
import constants as _constants_
import logging
from utils import utils
import json

import datetime

# 获取当前时间
current_time = datetime.datetime.now()

# 获取当前日期时间
now = datetime.datetime.now()

# 将日期时间转换为字符串
current_time = now.strftime("%Y-%m-%d %H:%M:%S")


# set training configurations
train_config = {
    'batch_size': 100,
    'num_epochs': 6,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 256,
    'eval_steps': 100,
    'save_steps': 100,
    # "save_path": save_model_path,
    "model_zoo": ""   # the path of offline models
}

transform = transforms.Compose([
              transforms.RandomHorizontalFlip(0.5),
              transforms.ColorJitter(0.2,0.2),
              transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
              transforms.Resize((256, 256)),
              transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
  torch.cuda.set_device(device)
  
# def print_configuration (merged_dict = merged_dict):
#   print (merged_dict)
#   if merged_dict["contrastive_param"] != 1:
#     print(f"contrastive loss parameter is {constants.RED + str(merged_dict["contrastive_param"]) + constants.RESET}")
#   if merged_dict['cls_param'] != 1:
#     print(f"classification loss parameter is {constants.RED + str(merged_dict['cls_param']) + constants.RESET}")
#   if merged_dict["orthogonal_param"] != 1:
#     print(f"orthogonal loss parameter is {constants.RED + str(merged_dict["orthogonal_param"]) + constants.RESET}")
#   if merged_dict["graph_param"] != 1:
#     print(f"high-order loss parameter is {constants.RED + str(merged_dict["graph_param"]) + constants.RESET}")
#   if merged_dict["trainable_PLM"] != 0:
#     print(f"the number of trainable layers is {constants.RED + str(merged_dict["trainable_PLM"]) + constants.RESET}")
#   if merged_dict["prompt"] != "basic":
#     print(f"the text prompt template is {constants.RED + merged_dict["prompt"] + constants.RESET}")
#   if  merged_dict["weight_strategy"] != "NA":
#     print(f"current weighting strategy is {constants.RED + merged_dict["weight_strategy"] + constants.RESET}")
#   if merged_dict["uncertain_based_weight"]:
#     print(constants.RED + "uning uncertain based strategy to weight different sublosses"+constants.RESET)
#   if merged_dict["two_phases"]:
#     print(constants.RED + "using two phase training scheme" + constants.RESET)
#   if merged_dict["no_orthogonize"]:
#     print(constants.RED + "do not implement orthogonization" + constants.RESET)
#   if merged_dict["no_contrastive"]:
#     print(constants.RED + "do not implement contrastive learning between text and images" + constants.RESET)
#   if merged_dict["learnable_weight"]:
#     print(constants.RED+"using learnable weights among sub-loss during training!"+constants.RESET)
#     logger.info("using learnable weights among sub-loss during training!")
#   if merged_dict["high_order"] != "NA":
#     print(constants.RED+f"integrate graph alignment into the whole loss, using {merged_dict["high_order"]} graph!"+constants.RESET)
#     logger.info(f"integrate graph alignment into the whole loss, using {merged_dict["high_order"]} graph!")
#   print(f"label_strategy setting -- {constants.RED} {merged_dict['labeling_strategy']} {constants.RESET}") 
#   return
  

def main():
  logger = utils.set_env_config()
  args_parser = utils.parser()
  args = args_parser.set_arg_parser()
  print(args)
  print()
  pwd = os.getcwd()
  utils.set_random_seed()

  num_of_thread = 1
  save_model_path = pwd + "/output/"

  num_workers = 5

  backbone = "biomedclip" if args.backbone == None else args.backbone
  backbone_v = None if args.backbone_v == None else args.backbone_v
  prompt = "basic" if args.prompt == None else args.prompt
  visual_branch_only = args.vision_only
  two_phases = args.two_phases
  uncertain_based_weight = args.uncertain_based_weight
  weight_strategy = args.weight_strategy
  no_contrastive = args.no_contrastive
  learnable_weight = args.learnable_weight
  high_order = args.high_order
  labeling_strategy = args.labeling_strategy
  no_orthogonal = args.no_orthogonal
  contrastive_param = args.contrastive_param
  cls_param = args.classification_param
  orthogonal_param = args.orthogonal_param
  graph_param = args.graph_param
  trainable_PLM = args.trainable_PLM
  AP_PA_view = args.AP_PA_view
  trainable_VisionEncoder = args.trainable_VisionEncoder
  Alignment_Only = args.Alignment_Only
  debug = args.debug
  
  tasks_configuration = {"no_contrastive" : args.no_contrastive,
                         "no_orthogonize" : args.no_orthogonal,
                         "high_order" : args.high_order, 
                         "contrastive_param": args.contrastive_param,
                         "cls_param" : args.classification_param,
                         "orthogonal_param" : args.orthogonal_param,
                         "graph_param" : args.graph_param,
                          "weight_strategy": weight_strategy,
                          "uncertain_based_weight": uncertain_based_weight,
                          "learnable_weight": learnable_weight
                         }
  
  samples_configuration = {"AP_PA_view" : AP_PA_view,
                        "labeling_strategy": args.labeling_strategy,
                        "prompt" : "basic" if args.prompt == None else args.prompt
                        }
  
  model_configuration = {"backbone" : "biomedclip" if args.backbone == None else args.backbone,
                          "backbone_v" : None if args.backbone_v == None else args.backbone_v,
                          "visual_branch_only" : args.vision_only,
                          "trainable_PLM" : args.trainable_PLM
                          }
  
  configurations = [
    "backbone","backbone_v", "visual_branch_only", "learnable_weight", "high_order", "no_orthogonize",
    "no_contrastive", "weight_strategy", "contrastive_param", "trainable_PLM", "prompt"
    ]
  
  merged_dict = {**tasks_configuration, **samples_configuration, **model_configuration}

  if args.Alignment_Only:
    save_model_path = os.path.join(save_model_path , "pretrained", current_time)
    
  elif args.save_dir == None:
    args_keys = [i for i in args.__dict__.keys()]
    args_values = [str(i) if not isinstance(i, str) else i for i in args.__dict__.values()]

    configuration_concat = '_'.join(args_values)
    save_model_path = os.path.join(save_model_path , configuration_concat)
    
    # save_model_path = (save_model_path +
    #            f"/{backbone}_{backbone_v}_{visual_branch_only}_{learnable_weight}_{high_order}_{no_orthogonize}_{no_contrastive}_{weight_strategy}_"
    #            f"{contrastive_param}_{trainable_PLM}_{prompt}/")
  
  else:
    save_model_path = save_model_path + "/" + args.save_dir
  
  print("saving path: ",save_model_path)
  if os.path.exists(save_model_path) : raise RuntimeError(f"{save_model_path} has already existed! Please double check.")
  if not os.path.exists(save_model_path): os.makedirs(save_model_path)
  with open(os.path.join(save_model_path, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)
  
  train_data = ImageTextContrastiveDataset(backbone_type=backbone, prompt_type = prompt, 
                                           labeling_strategy = labeling_strategy,
                                           AP_PA_view = AP_PA_view) 
  train_collate_fn = ImageTextContrastiveCollator()
  train_loader = DataLoader(train_data,
      batch_size=train_config['batch_size'],
      collate_fn=train_collate_fn,
      shuffle=True,
      # pin_memory=True,
      num_workers = num_workers,
      prefetch_factor = 5
      )
  
  param_dict = {"weight_strategy": uncertain_based_weight, "weighting_strategy": weight_strategy, 
                "contrastive_param": contrastive_param, "cls_param": cls_param,
                "orthogonal_param": orthogonal_param, "graph_param": graph_param,
                }
  train_dict = {"trainable_PLM": trainable_PLM,
                "trainable_VisionEncoder" : trainable_VisionEncoder,
                "Alignment_Only": Alignment_Only,
                }
  # model definition
  model = MultiTaskModel(nntype = backbone, visual_branch_only = visual_branch_only, backbone_v = backbone_v,high_order=high_order, 
                          no_orthogonal = no_orthogonal, no_contrastive=no_contrastive,labeling_strategy = labeling_strategy, 
                          **train_dict)
  # loss definition
  loss_model = LG_CLIP_LOSS(MultiTaskModel = model, learnable_weight=learnable_weight, **param_dict).to(device)
  total_trainable_params = utils.count_parameters(model)
  print(_constants_.RED + f"\nthe amount of trainable parameters is {total_trainable_params}.\n" + _constants_.RESET)
  # build evaluator
  val_data = TestingDataset(backbone_type=backbone, 
                            labeling_strategy = labeling_strategy,
                            AP_PA_view = AP_PA_view)
  val_collate_fn = TestingCollator()
  
  eval_dataloader = DataLoader(val_data,
      batch_size=train_config['eval_batch_size'],
      collate_fn=val_collate_fn,
      shuffle=False,
      # pin_memory=True,
      num_workers = num_workers,
      prefetch_factor = 5
      )
  _evaluator_ = Evaluator(
      FG_model_cls = model,
      eval_dataloader = eval_dataloader,
      labeling_strategy = labeling_strategy
      )

  # loss_model.cuda()
  train_objectives = [
      (train_loader, loss_model, 1),
  ]
  trainer = Trainer()

  try:
    # torch.autograd.set_detect_anomaly(True)
    # with torch.autograd.profiler.profile():     
    trainer.train(
      model, train_objectives= train_objectives, warmup_ratio=train_config['warmup'],
      epochs=train_config['num_epochs'], optimizer_params={'lr':train_config['lr']},
      output_path = save_model_path, evaluation_steps=train_config['eval_steps'],
      weight_decay=train_config['weight_decay'], save_steps=train_config['save_steps'],
      # steps_per_epoch = 1,
      evaluator = _evaluator_, eval_dataloader=eval_dataloader,
      use_amp=True, two_phases=two_phases, Alignment_Only = Alignment_Only, 
      debug = debug)
    print(_constants_.GREEN + 'done' + _constants_.RESET)
    # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", "retrain clip version (FG-CLIP_Vision_Branch_Only) done", "Success")
  except Exception as e:
    Traceback = traceback.format_exc()
    T = f"{Traceback}"
    # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", T, "error")
    print(T)
  
if __name__ == "__main__":
  main()


      
