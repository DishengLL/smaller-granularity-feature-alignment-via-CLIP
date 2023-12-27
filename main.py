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
import argparse
import traceback
import constants as _constants_
import logging



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
    logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('app.log')  # 输出到文件
        ]
    )

    # 创建日志记录器
    logger = logging.getLogger('my_logger')
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

    num_of_thread = 1
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
        'eval_steps': 100,
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
    parser.add_argument('--backbone', type=str,choices=["clip", "biomedclip"], help='the backbone module in the model')
    parser.add_argument('--prompt', type=str, help='the type of prompt used in the model training')
    parser.add_argument('--vision_only',action='store_true', default=False, help='does the model contain vision branch')
    parser.add_argument('--backbone_v', choices=['densenet'], type=str, help="vision encoder in image branch")
    parser.add_argument('--save_dir', type=str, help="the dir to save output")
    parser.add_argument('--learnable_weight',action='store_true', default=False, help='set learnable weights between differetn sub-losses(default: false)')
    parser.add_argument('--high_order',  type=str,choices=["binary", "KL_based", "NA"], default="NA", help='using high-order correlation contrastive learning during training(default: false)')
    parser.add_argument('--two_phases',action='store_true', default=False, help='implement 2-phases training scheme') 
    parser.add_argument('--no_orthogonize',action='store_true', default=False, help='do not implement orthogonization operation in the whole pipeline')
    parser.add_argument('--no_contrastive',action='store_true', default=False, help='do not implement contrastive alignment between text and images')  
    args = parser.parse_args()    
    backbone = "biomedclip" if args.backbone == None else args.backbone
    backbone_v = None if args.backbone_v == None else args.backbone_v
    prompt = "basic" if args.prompt == None else args.prompt
    visual_branch_only = args.vision_only
    two_phases = args.two_phases
    if two_phases:
      print(constants.RED + "using two phase training scheme" + constants.RESET)
    no_orthogonize = args.no_orthogonize
    if no_orthogonize:
      print(constants.RED + "do not implement orthogonization" + constants.RESET)
    no_contrastive = args.no_contrastive
    if no_contrastive:
      print(constants.RED + "do not implement contrastive learning between text and images" + constants.RESET)
    learnable_weight = args.learnable_weight
    high_order = args.high_order
    if learnable_weight:
      print(constants.RED+"using learnable weights among sub-loss during training!"+constants.RESET)
      logger.info("using learnable weights among sub-loss during training!")
    if high_order != "NA":
      print(constants.RED+f"integrate graph alignment into the whole loss, using {high_order} graph!"+constants.RESET)
      logger.info(f"integrate graph alignment into the whole loss, using {high_order} graph!")
    if  args.save_dir == None:
      save_model_path = save_model_path + f"/{backbone}_{backbone_v}_{visual_branch_only}/"
    else:
      save_model_path = save_model_path + "/" + args.save_dir
    print("saving path: ",save_model_path)
        
    train_data = ImageTextContrastiveDataset(backbone_type=backbone, prompt_type = prompt,) 
    train_collate_fn = ImageTextContrastiveCollator()
    train_loader = DataLoader(train_data,
        batch_size=train_config['batch_size'],
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers = num_of_thread,
        )

    model = MultiTaskModel(nntype = backbone, visual_branch_only = visual_branch_only, backbone_v = backbone_v, high_order=high_order, no_orthogonize = no_orthogonize)
    loss_model = LG_CLIP_LOSS(MultiTaskModel = model, learnable_weight=learnable_weight,).to(device)

    # build evaluator
    val_data = TestingDataset(backbone_type=backbone)
    val_collate_fn = TestingCollator()
    eval_dataloader = DataLoader(val_data,
        batch_size=train_config['eval_batch_size'],
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers = num_of_thread,
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
        use_amp=True,
        two_phases=two_phases)
      print(_constants_.GREEN + 'done' + _constants_.RESET)
      # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", "retrain clip version (FG-CLIP_Vision_Branch_Only) done", "Success")
    except Exception as e:
      Traceback = traceback.format_exc()
      T = f"{Traceback}"
      # email.send_email("1554200903@qq.com", f"train {backbone}-{backbone_v}-vision_only:{visual_branch_only}", T, "error")
      print(T)
      
