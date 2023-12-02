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
from _email_ import send_email
import traceback
import argparse

# def performance():
#     import cProfile
#     # Your code here
#       def run_trainer():
#         trainer.train(
#             model,
#             train_objectives=train_objectives,
#             warmup_ratio=train_config['warmup'],
#             epochs=train_config['num_epochs'],
#             optimizer_params={'lr': train_config['lr']},
#             output_path=train_config["save_path"],
#             evaluation_steps=train_config['eval_steps'],
#             weight_decay=train_config['weight_decay'],
#             save_steps=train_config['save_steps'],
#             evaluator=_evaluator_,
#             eval_dataloader=eval_dataloader,
#             use_amp=True,
#         )

#       cProfile.run('run_trainer()', r"D:\exchange\ShanghaiTech\learning\code\diagnosisP\x_ray_constrastive\profiler_stats")
#     except KeyboardInterrupt:
#         import sys
#         import pstats
#         print("Program terminated by user.")
#         stats = pstats.Stats(r"D:\exchange\ShanghaiTech\learning\code\diagnosisP\x_ray_constrastive\profiler_stats")

# # 按照执行时间降序排列并打印
#         stats.sort_stats('cumulative').print_stats()
#         # Additional cleanup or logging if needed
#         sys.exit(0)

if __name__ == "__main__":
    # print(constants.BLUE + f"run Fine-Grain Feature Alignment CLIP(FG_FA_C)" + constants.RESET)
    email = send_email.send_email()
    pwd = os.getcwd()
    # set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM']='false'

    num_of_thread = 4
    save_model_path = pwd + "/output/"

    # set cuda devices
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
      torch.cuda.set_device(device)

    # set training configurations
    train_config = {

        'batch_size': 100,
        'num_epochs': 10,
        'warmup': 0.1, # the first 10% of training steps are used for warm-up
        'lr': 2e-5,
        'weight_decay': 1e-4,
        'eval_batch_size': 256,
        'eval_steps': 1000,
        'save_steps': 1000,
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
    
    parser = argparse.ArgumentParser(description='parse input parameter for model configuration')
    parser.add_argument('--backbone', type=str, help='the backbone module in the model')
    parser.add_argument('--prompt', type=str, help='the type of prompt used in the model training')
    parser.add_argument('--vision_only', type=bool, default = False, help='does the model contain vision branch')
    parser.add_argument('--backbone_v', type=str, help="vision encoder in image branch")
    parser.add_argument('--save_dir', type=str, help="the dir to save output")
    args = parser.parse_args()    
    backbone = "biomedclip" if args.backbone == None else args.backbone
    backbone_v = None if args.backbone_v == None else args.backbone_v
    prompt = "basic" if args.prompt == None else args.prompt
    visual_branch_only = args.vision_only
    if  args.save_dir == None:
      save_model_path = save_model_path + f"/{backbone}_{backbone_v}_{visual_branch_only}/"
    else:
      save_model_path = save_model_path + "/" + args.save_dir
    print(">>>>>>>>>>",save_model_path)
        
    train_data = ImageTextContrastiveDataset(backbone_type=backbone, prompt_type = prompt,) 
    train_collate_fn = ImageTextContrastiveCollator()
    train_loader = DataLoader(train_data,
        batch_size=train_config['batch_size'],
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers = num_of_thread,
        )

    model = MultiTaskModel(nntype = backbone, visual_branch_only = visual_branch_only, backbone_v = backbone_v)
    loss_model = LG_CLIP_LOSS(MultiTaskModel = model).to(device)

    # build evaluator
    val_data = TestingDataset(backbone_type=backbone)
    val_collate_fn = TestingCollator()
    eval_dataloader = DataLoader(val_data,
        batch_size=train_config['eval_batch_size'],
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers = 4,
        )
    _evaluator_ = Evaluator(
        FG_model_cls = model,
        eval_dataloader = eval_dataloader,
        mode='multiclass')

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
        use_amp=True,)
      print('done')
      # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", "retrain clip version (FG-CLIP_Vision_Branch_Only) done", "Success")
    except Exception as e:
      Traceback = traceback.format_exc()
      T = f"{Traceback}"
      # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", T, "error")
      print(T)
      
