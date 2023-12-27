from calendar import c
import os
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers
import constants as _constants_

WEIGHTS_NAME = "pytorch_model.bin"
pwd = os.getcwd()


class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader = None,
        evaluator=None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 100,
        save_steps : int = 100,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_total_limit: int = 0,
        load_best_model_at_last: bool = True,
        two_phases = False
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        self.best_auc = -9999999
        self.accumulation_steps = accumulation_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.score_logs = defaultdict(list)
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader
        dataloaders = [dataloader for dataloader,_,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        loss_models = [loss for _, loss,_ in train_objectives]
        train_weights = [weight for _,_,weight in train_objectives]

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # map models to devices
        model = model.to(self.device)

        # execute training on multiple GPUs
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        train_loss_dict = defaultdict(list)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            # print("steps_per_epoch>>", steps_per_epoch)
            for train_iter in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable= not show_progress_bar): # the number of batches

                # check if model parameters keep same
                for train_idx in range(num_train_objectives): # calculate for each train objective 
                    loss_model = loss_models[train_idx]
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_weight = train_weights[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                        
                    except StopIteration:
                        # raise NotImplementedError("Something wrong happens")
                        if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                            dataloaders[train_idx].dataset._build_prompt_sentence()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)
                    if use_amp:
                        with autocast():
                            loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return#['loss_value']
                        loss_value = loss_value
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        # print(data)
                        loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and global_step % evaluation_steps == 0:
                    print('\n######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{}: {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                    #TODO: update prompt sentences
                    # for train_idx in range(num_train_objectives):
                    #     if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                    #         dataloaders[train_idx].dataset._build_prompt_sentence()

                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f'\n\033[31m######### Eval {global_step} #########\033[0m')
                    for key in scores.keys():
                        if key in ['acc','auc', 'auc/mse', ]:
                          print('{}: {:.4f}'.format(key, scores[key]))
                        if key == "auc_dict":
                          for i,j in scores[key].items():
                            print(i, j)
                          av_auc = get_average_auc_among_disease(auc_dict, indicator = "positive")
                          if av_auc > self.best_auc:
                            self.best_auc = av_auc
                            print(f"update best avg auc : {self.best_auc}")
                            save_dir = os.path.join(output_path,"")
                            self._save_ckpt(model, save_dir)     #save the best model during the iterations
                    print(_constants_.GREEN + f"the classifier loss: {scores['loss']}" + _constants_.RESET)
                    print(f'\n\033[31m#######################################\033[0m')
                          # print("auc_dict: \n", scores[key])
                    # save_dir = os.path.join(output_path, f'{global_step}/')
                    # self._save_ckpt(model, save_dir)

                    # score logs save the list of scores
                    self.score_logs['global_step'].append(global_step)
                    for key in scores.keys():
                        if key in ['acc','auc', 'auc/mse']:
                            self.score_logs[key].append(scores[key])

                if self.evaluator is None and global_step % save_steps == 0:
                    state_dict = model.state_dict()
                    save_dir =  os.path.join(output_path, f'{global_step}/')
                    # self._save_ckpt(model, save_dir)
                    # print('model saved to', os.path.join(output_path, WEIGHTS_NAME))
                    
                if torch.cuda.is_available():
                  torch.cuda.empty_cache()

        # if save_best_model:
        #     import pandas as pd
        #     from distutils.dir_util import copy_tree
        #     res = pd.DataFrame(self.score_logs)
        #     res.to_csv(output_path + r"/res.csv")
        #     res = res.set_index('global_step')
            # take the average column best
            # print(res.mean(1))
            # best_iter = res.mean(1).idxmax()
            # best_save_path = os.path.join(output_path, './best')
            # if not os.path.exists(best_save_path): os.makedirs(best_save_path)
            # best_origin_path = os.path.join(output_path, f'./{best_iter}')
            # print(f'save best checkpoint at iter {best_iter} to', best_save_path)
            # try:
            #   copy_tree(best_origin_path, best_save_path)
            # except:
            #     print(_constants_.RED + "copy_tree error in main.py" + _constants_.RESET)
                

        if eval_dataloader is None and output_path is not None:   #No evaluator, but output path: save final model version
            state_dict = model.state_dict()
            if "/Users/liu/Desktop/school_academy/ShanghaiTech" in output_path:
                output_path = output_path.replace("/Users/liu/Desktop/school_academy/ShanghaiTech", "D://exchange//ShanghaiTech//")
            # torch.save(state_dict, os.path.join(output_path, WEIGHTS_NAME))
            # print('model saved to', os.path.join(output_path, WEIGHTS_NAME))
            # torch.save(model,  os.path.join(output_path, "whole_model.pth"))

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        state_dict = model.state_dict()
        print("save_dir: ", save_dir)
        torch.save(state_dict, os.path.join(save_dir, WEIGHTS_NAME))
        # torch.save(model, os.path.join(save_dir, "model.pt"))

    def _export_onnx_(self,model, onnx_file_path):
        # onnx_file_path = "your_model.onnx"  # 保存的 ONNX 文件名

        # 使用 torch.onnx.export 导出模型
        torch.onnx.export(model,  # 已加载的 PyTorch 模型
                        torch.randn(1, 1) ,  # 示例输入数据
                        onnx_file_path,  # 保存的 ONNX 文件路径
                        verbose=True,  # 可选参数，用于显示详细信息
                        # input_names=['input'],  # 指定输入张量的名称
                        # output_names=['output'])  # 指定输出张量的名称
                        )   
        
    def get_average_auc_among_disease(auc_dict, indicator = "positive"):
      average_auc = 0
      n_disease = len(auc_dict)
      for disease, auc in auc_dict.items():
        v = auc[indicator]
        average_auc = average_auc + v
      return average_auc/n_disease
        
      