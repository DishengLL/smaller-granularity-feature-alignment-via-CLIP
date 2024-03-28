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
  

def main():
  logger = utils.set_env_config()
  args_parser = utils.parser()
  args = args_parser.set_arg_parser()
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
  no_orthogonize = args.no_orthogonize
  
  if  weight_strategy != "NA":
    print(f"current weighting strategy is {constants.RED + weight_strategy + constants.RESET}")
  if uncertain_based_weight:
    print(constants.RED + "uning uncertain based strategy to weight different sublosses"+constants.RESET)
  if two_phases:
    print(constants.RED + "using two phase training scheme" + constants.RESET)
  if no_orthogonize:
    print(constants.RED + "do not implement orthogonization" + constants.RESET)
  if no_contrastive:
    print(constants.RED + "do not implement contrastive learning between text and images" + constants.RESET)
  if learnable_weight:
    print(constants.RED+"using learnable weights among sub-loss during training!"+constants.RESET)
    logger.info("using learnable weights among sub-loss during training!")
  if high_order != "NA":
    print(constants.RED+f"integrate graph alignment into the whole loss, using {high_order} graph!"+constants.RESET)
    logger.info(f"integrate graph alignment into the whole loss, using {high_order} graph!")
  if  args.save_dir == None:
    save_model_path = save_model_path + f"/{backbone}_{backbone_v}_{visual_branch_only}_{learnable_weight}_{high_order}_{no_orthogonize}_{no_contrastive}_{weight_strategy}/"
  else:
    save_model_path = save_model_path + "/" + args.save_dir
  
  print("saving path: ",save_model_path)
  print(f"label_strategy setting -- {constants.RED} {labeling_strategy} {constants.RESET}") 
  
  train_data = ImageTextContrastiveDataset(backbone_type=backbone, prompt_type = prompt, labeling_strategy = labeling_strategy) 
  train_collate_fn = ImageTextContrastiveCollator()
  train_loader = DataLoader(train_data,
      batch_size=train_config['batch_size'],
      collate_fn=train_collate_fn,
      shuffle=True,
      # pin_memory=True,
      num_workers = num_workers,
      prefetch_factor = 5
      )
  param_dict = {"weight_strategy": uncertain_based_weight, "weighting_strategy": weight_strategy}
  
  # model definition
  model = MultiTaskModel(nntype = backbone, visual_branch_only = visual_branch_only, backbone_v = backbone_v,high_order=high_order, 
                          no_orthogonize = no_orthogonize, no_contrastive=no_contrastive,labeling_strategy = labeling_strategy)
  # loss definition
  loss_model = LG_CLIP_LOSS(MultiTaskModel = model, learnable_weight=learnable_weight, **param_dict).to(device)

  # build evaluator
  val_data = TestingDataset(backbone_type=backbone, labeling_strategy = labeling_strategy)
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
      model,
      train_objectives= train_objectives,
      warmup_ratio=train_config['warmup'],
      epochs=train_config['num_epochs'],
      optimizer_params={'lr':train_config['lr']},
      output_path = save_model_path,
      evaluation_steps=train_config['eval_steps'],
      weight_decay=train_config['weight_decay'],
      save_steps=train_config['save_steps'],
      # steps_per_epoch = 1,
      evaluator = _evaluator_,
      eval_dataloader=eval_dataloader,
      use_amp=True,
      two_phases=two_phases)
    print(_constants_.GREEN + 'done' + _constants_.RESET)
    # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", "retrain clip version (FG-CLIP_Vision_Branch_Only) done", "Success")
  except Exception as e:
    Traceback = traceback.format_exc()
    T = f"{Traceback}"
    # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", T, "error")
    print(T)
  
if __name__ == "__main__":
  main()


      
