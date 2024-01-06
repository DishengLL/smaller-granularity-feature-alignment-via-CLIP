import torch
from models import MultiTaskModel   
import constants as _constants_
import  matplotlib.pyplot as plt
# import umap
import numpy as np
from PIL import Image
import clip
from torch.utils.data import DataLoader
from dataset import TestingCollator, TestingDataset
from evaluate import  Evaluator
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
import argparse
import traceback
import constants as _constants_
import logging
import os


def plot(get_text_embedding, prompt, title = None):
  text_features = get_text_embedding
  similarity_matrix = torch.mm(text_features, text_features.t())
  text_features = text_features / text_features.norm(dim=-1, keepdim=True)
  # text_features = text_features.detach().numpy()
  similarity = (1 * text_features @ text_features.T)
  print(similarity.shape)
  similarity=similarity.cpu().detach().numpy()
  # data = similarity_matrix.cpu().numpy()
  # data.shape

  plt.figure(figsize=(8, 5))
  plt.imshow(similarity)
  plt.colorbar()
  plt.yticks(range(len(_constants_.CHEXPERT_LABELS)), prompt, fontsize=7)
  plt.xticks(range(len(_constants_.CHEXPERT_LABELS)), prompt, fontsize=7, rotation=40)
  # for i, image in enumerate(text):
  #     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
  for x in range(similarity.shape[1]):
      for y in range(similarity.shape[0]):
          plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=8)

  for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

  # plt.xlim([-0.5, 4 - 0.5])
  # plt.ylim([4 + 0.5, -2])
  print(len(similarity))
  avg_sim = (np.sum(similarity) - len(similarity))/(len(similarity)*(len(similarity)-1))
  title = "Cosine similarity between text and text features CLIP TEXT encoder" if title is None else title
  plt.title(title + f" : avg_sim: {avg_sim}", size=20)
  # plot.show()

def load_model_vis_embedding():
  model = MultiTaskModel(nntype="biomedclip")

  # the model class: 你所定义的模型的class
  model.load_state_dict(torch.load(r"D:\exchange\ShanghaiTech\learning\code\diagnosisP\x_ray_constrastive\output\checkpoint_11_11_bio\best\pytorch_model.bin"))
  (model.eval())

  prompt = _constants_.BASIC_PROMPT
  image_path = ["D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\imgs\\structure.png"]
  label = torch.tensor([1,1,2,1,2,1,1,2,1,0,0,2,1])
  model.cuda()
  contrastive_mode, classifier, orthogonal = model(prompt, image_path, label)
  print(contrastive_mode.keys(), _constants_.RED + "constrastive_loss: "+str(contrastive_mode["loss_value"])+_constants_.RESET)
  print(classifier.keys() ,_constants_.RED + "classifier_loss:"+str(classifier["loss_value"])+_constants_.RESET)
  print(orthogonal.keys() , _constants_.RED + "orthogonal_loss: "+str(orthogonal["loss_value"])+ _constants_.RESET)

  text_embedding = orthogonal['text_embeds']
  print(text_embedding.shape)
  text_embedding = text_embedding.squeeze()
  print(text_embedding.shape)

  reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
  embedding = reducer.fit_transform(text_embedding.cpu().detach().numpy())

  # 提取降维后的坐标
  x = embedding[:, 0]
  y = embedding[:, 1]
  labels = _constants_.BASIC_PROMPT

  plt.scatter(x, y)
  for i in range(len(labels)):
      plt.annotate(labels[i], (x[i], y[i]))  # 在点旁边添加标签

  plt.title("UMAP Projection of Data with Labels using the data of trained model")
  plt.grid()

  plot(text_embedding.cpu(),prompt, "sim between text embedding generated by FG+CLIP")
  plt.show()

class print_plot_CLIP():
  def __init__(self):
    self.prompt = _constants_.BASIC_PROMPT

  def get_text_embedding(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in self.prompt]).to(device)

    # Calculate features
    print(text_inputs.shape)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    print(text_features.size())
    return text_features

  def plot(self):
    text_features = self.get_text_embedding()
    similarity_matrix = torch.mm(text_features, text_features.t())
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (1 * text_features @ text_features.T)
    print(similarity.shape)
    similarity=similarity.cpu().numpy()
    data = similarity_matrix.cpu().numpy()
    data.shape

    plt.figure(figsize=(8, 5))
    plt.imshow(similarity)
    plt.colorbar()
    plt.yticks(range(13), self.prompt, fontsize=7)
    plt.xticks(range(13), self.prompt, fontsize=7, rotation=40)
    # for i, image in enumerate(text):
    #     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=8)

    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)

    # plt.xlim([-0.5, 4 - 0.5])
    # plt.ylim([4 + 0.5, -2])

    plt.title("Cosine similarity between text and text features CLIP TEXT encoder", size=20)
    similarity

  def plot_UMAP(self):
    text_features = self.get_text_embedding()
    random_seed = 42
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3, random_state=random_seed)
    from sklearn.preprocessing import StandardScaler  # 导入标准化工具
    scaler = StandardScaler()
    text_embedding = scaler.fit_transform(text_features.cpu().detach().numpy())

    embedding = reducer.fit_transform(text_embedding)

    # 计算均值
    mean_x = np.mean(embedding[:, 0])
    mean_y = np.mean(embedding[:, 1])
    mean_z = np.mean(embedding[:, 2])

    # 平移所有点，使原点位于中心
    embedding_centered = embedding - [mean_x, mean_y, mean_z]

    fig = plt.figure(figsize=(19, 13))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("UMAP in original CLIP text embedding")

    # 使用 ax.quiver 绘制三维向量
    for i in range(len(embedding)):
        x_start = 0  # 起点 x 坐标
        y_start = 0  # 起点 y 坐标
        z_start = 0  # 起点 z 坐标

        x_vector = embedding_centered[i, 0]  # x 方向上的分量
        y_vector = embedding_centered[i, 1]  # y 方向上的分量
        z_vector = embedding_centered[i, 2]  # z 方向上的分量

        # ax.quiver(x_start, y_start, z_start, x_vector, y_vector, z_vector)
        color = plt.cm.viridis((i) / len(embedding_centered))  # 根据索引设置颜色
        ax.quiver(x_start, y_start, z_start, x_vector, y_vector, z_vector, color=color, label=self.prompt[i])

        print(x_vector, y_vector, z_vector)

        # ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], prompt[i])
    max_range = np.max(np.abs(embedding_centered))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range ])
    ax.set_zlim([-max_range, max_range])
    ax.legend()
    # plt.grid()
    plt.show()
   
# a = print_plot_CLIP()
# a.plot()
# a.plot_UMAP()
        
def get_average_auc_among_disease( auc_dict, indicator = "positive"):
  average_auc = 0
  n_disease = len(auc_dict)
  for disease, auc in auc_dict.items():
    v = auc[indicator]
    average_auc = average_auc + v
  return average_auc/n_disease
        
      

def load_model(path = None, nntype = 'biomedclip' ):
  if path == None:
    raise ValueError("you should specify the path of model")
  model = MultiTaskModel(nntype="biomedclip")
  model.load_state_dict(torch.load(path))
  model.eval()
  return model

def get_AUC(predictions_tensor, labels_tensor, plot=False, record_roc = False, training_step = 0):
  """
  for each label(disease) gets its own auc
  plot: bool, whether plot the roc plot in this function
  return auc value 
  """
  disease_auc = {}
  bins = [i/20 for i in range(20)] + [1]
  for i, disease in enumerate(constants.CHEXPERT_LABELS):
    label_dis = labels_tensor[:, i]
    each_class_roc = {}
    for k, j in enumerate(constants.class_name):
      pred_dis = predictions_tensor[:, i*len(constants.class_name) + k].cpu().numpy()
      true_class = [1 if constants.class_name[j] == y else 0 for y in label_dis]
      if(len(set(true_class))==1):
        print(constants.RED, "this disease have something wrong: "+constants.RESET, disease, ", ", j, "in this case set auc is 0!!!")
        each_class_roc[j] = 0
        continue
      # self.plot_(true_class, pred_dis)
      each_class_roc[j] = roc_auc_score(true_class, pred_dis, multi_class="ovr", average="micro",)
    disease_auc[disease] = each_class_roc
  return disease_auc

def model_infer_eval(model = None):
  if model is None:
    raise ValueError("you should specify the model before inference")
    # build evaluator
  model.cuda()
  
  val_data = TestingDataset(backbone_type="biomedclip")
  val_collate_fn = TestingCollator()
  eval_dataloader = DataLoader(val_data,
      batch_size=256,
      collate_fn=val_collate_fn,
      shuffle=False,
      pin_memory=True,
      num_workers = 2,
      )
  _evaluator_ = Evaluator(
      FG_model_cls = model,
      eval_dataloader = eval_dataloader,
      mode='multiclass')
    
  dump = {
    "prediction":"./output/dump/prediction/",
    "label":"./output/dump/labels/",
    "dump_path":"/output/dump/bio_high_task/"
  }
  scores = _evaluator_.evaluate(dump = dump)
  print(f'\n\033[31m######### Eval #########\033[0m')
  for key in scores.keys():
      if key in ['acc','auc', 'auc/mse', ]:
        print('{}: {:.4f}'.format(key, scores[key]))
      if key == "auc_dict":
        for i,j in scores[key].items():
          print(i, j)
        av_auc = get_average_auc_among_disease(scores[key], indicator = "positive")

          
  print(_constants_.GREEN + f"the classifier loss: {scores['loss']}" + _constants_.RESET)
  print(f'\n\033[31m#######################################\033[0m')
  # print(contrastive_mode.keys(), _constants_.RED + "constrastive_loss: "+str(contrastive_mode["loss_value"])+_constants_.RESET)
  # print(classifier.keys() ,_constants_.RED + "classifier_loss:"+str(classifier["loss_value"])+_constants_.RESET)
  # print(orthogonal.keys() , _constants_.RED + "orthogonal_loss: "+str(orthogonal["loss_value"])+ _constants_.RESET)
  # contrastive_mode, classifier, orthogonal = model(prompt, image_path, label)
  

def get_auc_roc(model_path = None):
  model = load_model(model_path)


if __name__ == "__main__":
  model = load_model("./output/biomedclip_None_False_False_binary_False_False_task_balance/pytorch_model.bin")
  model = load_model("/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/output/biomedclip_None_False_False_binary_False_False_task_balance/final_pytorch_model.bin")
  model_infer_eval(model)
