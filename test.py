from calendar import c
import os
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math
from networkx import katz_centrality_numpy

import numpy as np
from scipy import constants
from sympy import EX, false
import torch
from torch import nn
from torch import device, Tensor
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers
import constants as _constants_

from models import MultiTaskModel
from dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, TestingCollator, TestingDataset
from evaluate import  Evaluator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
WEIGHTS_NAME = "pytorch_model.bin"
import pickle

from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.special import softmax

def get_multiclass_auc(labels, predictions, key):
  classes = _constants_.class_name
  diseases = _constants_.CHEXPERT_LABELS
  # print(_constants_.RED, labels.shape, _constants_.RESET)
  # print(_constants_.RED, predictions.shape, _constants_.RESET)

  auc_dict = {}
  auc_disease = {}

  #AUC for each class -- positive, negative, uncertain
  for class_index in range(predictions.shape[1]):
      # 提取当前类别的真实标签和logits
      true_labels = labels[:, class_index]
      logits = predictions[:, class_index, :]
      softmax_probs = softmax(logits, axis=1)
      # true_labels = np.eye(3)[true_labels]
      # print(true_labels.shape)
      try:
        auc_score = roc_auc_score(true_labels, softmax_probs, multi_class='ovr',)
      except Exception as e:
          # print(diseases[class_index], "does not have 3 classes")
          continue
          logits = logits[:, (0,1)]
          softmax_probs = softmax(logits, axis=1)
          # print(softmax_probs.shape)
          true_labels[true_labels == 2] = 1
          auc_score = roc_auc_score(true_labels, softmax_probs[:,1],)
          auc_disease[f'{diseases[class_index]}'] = auc_score
          continue
      # 存储 AUC 值到字典
      auc_disease[f'{diseases[class_index]}'] = auc_score

  # 计算平均 AUC
  average_auc = np.mean(list(auc_disease.values()))
  auc_disease["average"] = average_auc
  auc_dict["class"] = auc_disease

  # 打印每个类别的 AUC 和平均 AUC
  for class_key, class_auc in auc_disease.items():
      print(f'AUC for {class_key}: {class_auc:.4f}')

  store = r"/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/output/temp_var"
  with open(f"{store}\{key}_without_no_finding", 'wb') as f:
    pickle.dump(auc_disease, f)


def process_confusion_matrix(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    outputs = {}
    # Sensitivity, hit rate, recall, or true positive rate
    outputs['tpr'] = TP/(TP+FN)
    # Specificity or true negative rate
    outputs['tnr'] = TN/(TN+FP) 
    # Precision or positive predictive value
    outputs['ppv'] = TP/(TP+FP)
    # Negative predictive value
    outputs['npv'] = TN/(TN+FN)
    # Fall out or false positive rate
    outputs['fpr'] = FP/(FP+TN)
    # False negative rate
    outputs['fnr'] = FN/(TP+FN)
    # False discovery rate
    outputs['fdr'] = FP/(TP+FP)

    # Overall accuracy for each class
    # outputs['acc'] = (TP+TN)/(TP+FP+FN+TN)
    if cnf_matrix.shape[0] > 2: # multiclass
        for k,v in outputs.items(): # take macro avg over each class
            outputs[k] = np.mean(v)
    else:
        for k,v in outputs.items(): # take macro avg over each class
            outputs[k] = v[1]
    return outputs


def get_confusion(results, key):
    overall_logits  = results['overall_logit']  # z x batch x 13 x class number
    overall_predic_label = results["overall_prediction"] # z x batch_size x 13
    overall_labels = results["overall_label"] # z x batch_size x 13
    classname = _constants_.CHEXPERT_LABELS
    # print(overall_prediction)
    # print(overall_prediction[0].shape)
    # print("overall_predic_label:\n", overall_predic_label)
    # print("overall_labels:\n", overall_predic_label)
    n = len(classname)
    predict_labels = np.split(overall_predic_label, overall_predic_label.shape[1], axis=1)
    label = np.split(overall_labels, overall_labels.shape[1], axis=1)
    # for i, (predict, lebel) in enumerate(zip(predict_labels, label)):
        # predict = predict.squeeze()
        # label = label.squeeze()
        # cnf_matrix = confusion_matrix(lebel, predict)
        # print(cnf_matrix)c
        # res =process_confusion_matrix(cnf_matrix)
        # print(res)
    aucs = get_multiclass_auc(overall_labels, overall_logits, key)
    return

def get_testing_results(batch_size = 5, vision_only = None, nntype = None):
    model_dict = {
                  "biomedclip - vision-only":"/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/models/Biomed_Vision/best/pytorch_model.bin",
                  "biomedclip": "/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/models/Biomed/best/pytorch_model.bin",
                  "clip - vision-only": "/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/models/CLIP_Vision/best/pytorch_model.bin",
                  "clip": "coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/models/CLIP/pytorch_model.bin"
      }
    
    if vision_only:
      key = f"{nntype}"+" - vision-only"
    else:
      key = f"{nntype}"
    model = MultiTaskModel(nntype=nntype, visual_branch_only=vision_only)#.to(device)
    path = model_dict[key]
    print(f"load model from {path}")
    model.load_state_dict(torch.load(path))
    # prompt = _constants_.BASIC_PROMPT
    model.eval()
    model.cuda()
    val_data = TestingDataset()
    val_collate_fn = TestingCollator()
    eval_dataloader = DataLoader(val_data,
    batch_size=batch_size,
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers = 4,
    )
    _evaluator_ = Evaluator(
        FG_model_cls = model,
        eval_dataloader = eval_dataloader,
        mode='multiclass')

    scores = _evaluator_.evaluate_testing()
    return scores, key

def plot_test_acc_distribution(accs, key):
    plt.figure(figsize=(8, 6))
    acc_list = [i['acc'] for i in accs]
    # print(acc_list)
    mu, std = norm.fit(acc_list)
    
    sns.histplot(acc_list, kde=True, color='skyblue', stat='density')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Data Distribution Plot with Fitted Normal Distribution')

    # Annotate with mean and standard deviation
    plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f}')
    plt.axvline(mu + std, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std:.2f}')
    plt.axvline(mu - std, color='green', linestyle='dashed', linewidth=2)

    # Show legend
    plt.legend()

    # Show the plot

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(f'Data Distribution Plot in {key}')
    plt.grid()
    plt.savefig(f'/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/output/out/{key}.png')
    

def get_confusion_matrxi():
    return
    
def get_auc(get_save_roc=False):
    return
    

if __name__ == "__main__":
          nntype = "biomedclip" #biomedclip
          vision_only = True

          for nntype in ["clip", "biomedclip"]:
            for vision_only in [True, False]:
              results, key = get_testing_results(vision_only = vision_only, nntype = nntype)
              # plot_test_acc_distribution(results['result'], key)
              get_confusion(results, key)
