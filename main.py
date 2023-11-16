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


if __name__ == "__main__":
    print(f"run Fine-Grain Feature Alignment CLIP(FG_FA_C)")
    email = send_email.send_email()
    # set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    num_of_thread = 8
    save_model_path = "./code/diagnosisP/x_ray_constrastive/output/checkopint/"

    # set cuda devices
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        "save_path": "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\output\\checkpoint"
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

    train_data = ImageTextContrastiveDataset()
    train_collate_fn = ImageTextContrastiveCollator()
    train_loader = DataLoader(train_data,
        batch_size=train_config['batch_size'],
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers = num_of_thread,
        )

    model = MultiTaskModel(nntype="clip", visual_branch_only=True)#.to(device)
    loss_model = LG_CLIP_LOSS(MultiTaskModel = model).to(device)

    # build evaluator
    val_data = TestingDataset()
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
    model_save_path = save_model_path
    trainer = Trainer()

    try:

#       import cProfile
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
      trainer.train(
          model,
          train_objectives= train_objectives,
          warmup_ratio=train_config['warmup'],
          epochs=train_config['num_epochs'],
          optimizer_params={'lr':train_config['lr']},
          output_path=train_config["save_path"],
          evaluation_steps=train_config['eval_steps'],
          weight_decay=train_config['weight_decay'],
          save_steps=train_config['save_steps'],
          # steps_per_epoch = 1,
          evaluator = _evaluator_,
          eval_dataloader=eval_dataloader,
          use_amp=True,
          )
      print('done')
      email.send_email("1554200903@qq.com", "train FG-CLIP_Vision_branch_only", "retrain clip version (FG-CLIP_Vision_Branch_Only) done", "Success")
    except Exception as e:
      Traceback = traceback.format_exc()
      T = f"{Traceback}"
      email.send_email("1554200903@qq.com", "train FG-CLIP_Vision_branch_only", T, "error")
      print(T)
      
