import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from typing import List
import logging

logger = logging.getLogger(__name__)
## Loss optimizes parameters in Text_encoder -- done
class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss,self).__init__()

    def forward(self, vectors):
        # 计算余弦相似度矩阵
        cosine_similarity = self.cosine_similarity_with_norm(vectors)
        
        # 获取对角线上的元素
        diagonal = torch.diagonal(cosine_similarity)
        
        # 排除对角线上的相似度，将其设置为0
        cosine_similarity = cosine_similarity - torch.diag(diagonal)
        cosine_similarity = torch.abs(cosine_similarity)
        
        # 计算损失，按公式 L_{MOL} = 1/2 * sum(CosineSimilarity)
        loss = 0.5 * torch.sum(cosine_similarity) #/ (vectors.size(0) * (vectors.size(0) - 1))

        return loss

    def cosine_similarity_with_norm(self, vectors):
        # 计算点积
        dot_product = torch.matmul(vectors, vectors.t())
        
        # 计算向量长度（L2范数）
        norm = torch.norm(vectors, dim=1, keepdim=True)
        
        # 计算余弦相似度
        similarity = dot_product / (norm * norm.t())
        
        return similarity

## Loss optimizes parameters in Vis_encoder + Text_encoder
class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # Model = text_encoder + vis_encoder, output --> cosine similarity between T and I
    def forward(self,
        text_input = None,  # CLIP text_embeddings output 
        img_input = None, # CLIP vis_embeddings output
        img_labels = None,  # predefine diseases [0, 1, 2, 3, 4, ...]
        text_labels = None, # predefine diseases [0, 1, 2, 3, 4, ...]
        **kwargs,
        ):
        '''args:
        img_labels: the image corresponds to which classes of diagnoses
        text_labels: the text corresponds to which classes of diagnoses
        '''
        if img_labels is None or text_labels is None:
            raise NotImplementedError("labels should be given!")
        else:
            '''use soft clip loss
            '''
            outputs = self.model(
                    img = img_input,
                    text = text_input,
                    return_loss=False,
                    )

            # get logits
            logits = outputs['logits']  # which should contains the similarity matrix of texts and images

            # compute soft-labels, -1: negative, 0: uncertain, 1: positive
            # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
            label_sim = torch.matmul(img_labels, text_labels.T)
            label_sim = label_sim.to(logits.device)
            outputs['loss_value'] = self.contrastive(logits, label_sim)
        return outputs

    def contrastive(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(logits_per_img, F.softmax(soft_label,1))
        caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        return (image_loss + caption_loss) / 2

        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        # return image_loss

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]

    def _soft_bce_loss(self, input, target):
        return nn.functional.binary_cross_entropy_with_logits(input, target)

## Loss optimizes Parameters in Vis_encoder + Classifier -- done
class WCE_Loss(nn.Module):
    def __init__(self, model=None, loss_fn=None):
        super().__init__()
        # self.model = model  # Vis_encoder + Classifier
        # self.custom = True if loss_fn == None else False
        # if model.mode is None:
        #     raise NotImplementedError(f"take care model mode setting, if it has not been set, the default is multi-label\nCurrent, the program will raise ERROR!!!")
        # self.mode = model.mode
        print('in this function, we define Weighted cross entropy loss')

    def forward(self,
        logits = None,
        labels=None,
        **kwargs):
        # compute soft-labels, -1: negative, 0: uncertain, 1: positive
        # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
        #apply weighted BCE loss in each batch
        if isinstance(labels, torch.Tensor):
          p_n = labels.sum(axis=0)  # positive number
          n_n = labels.shape[0] - p_n
          p_w = (p_n + n_n + 1) / (p_n + 1)
          n_w = (p_n + n_n + 1) / (n_n + 1)

          assert p_w.shape[0] == n_w.shape[0] == labels.shape[-1], f"the shape of weights is not consistent with the shape of labels"
          self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = p_w)
          print("training with weighted cross-entropy loss in image classification")
      
        loss = self.loss_fn(logits, labels)
        return loss

class LG_CLIP_LOSS(nn.Module):
    def __init__(self,  alpha = 1, beta = 1, gamma = 1, delta = 1, MultiTaskModel=None, 
                 learnable_weight = False,  **kwargs,):
        super().__init__()
        if learnable_weight: 
          self.alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
          self.beta = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)
          self.gamma = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
          self.delta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
          self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
          self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
          self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
          self.delta = nn.Parameter(torch.tensor(delta), requires_grad=False)
        weighting_strategy = kwargs["weighting_strategy"]
        if weighting_strategy == "uncertain_based_weight":   ## 优先于learning weight 参数   
          self.alpha = torch.nn.Parameter(torch.abs(torch.randn(1)), requires_grad=True)
          self.beta = torch.nn.Parameter(torch.abs(torch.randn(1,)),requires_grad=True)
          self.gamma = torch.nn.Parameter(torch.abs(torch.randn(1)), requires_grad=True)
          self.delta = torch.nn.Parameter(torch.abs(torch.randn(1)), requires_grad=True)
        elif weighting_strategy == "task_balance":
          logger.info("I using monotonical increasing function: $e^(x)$")
          
        self.weighting_strategy = weighting_strategy
        if MultiTaskModel is None:
            raise ValueError("input MultiTaskModel is None!!!!")
        self.model = MultiTaskModel
        self.learnable_weight = learnable_weight
      
        if kwargs["contrastive_param"] != 1:
          ### specify the weight of contrastive loss, in this case, the weight of contrastive loss is fixed.
          self.alpha = nn.Parameter(torch.tensor(kwargs["contrastive_param"]), requires_grad=False)
        if kwargs["cls_param"] != 1:
          ### specify the weight of classification loss, in this case, the weight of contrastive loss is fixed.
          self.beta = nn.Parameter(torch.tensor(kwargs["cls_param"]), requires_grad=False)
        if kwargs["orthogonal_param"] != 1:
          ### specify the weight of orthogonal loss, in this case, the weight of contrastive loss is fixed.
          self.gamma = nn.Parameter(torch.tensor(kwargs["orthogonal_param"]), requires_grad=False)
        if kwargs["graph_param"] != 1:
          ### specify the weight of high-order loss, in this case, the weight of contrastive loss is fixed.
          self.delta = nn.Parameter(torch.tensor(kwargs["graph_param"]), requires_grad=False)

    def forward(self, 
                img = None,
                prompts:list = None,
                img_labels = None):
        if img is None:
            raise ValueError("input image_path is None")
        if prompts is None:
            raise ValueError("input prompts is None")
        if img_labels is None:
            raise ValueError("img_label which will be used in Classifier is None")
        FGA, Cls = self.model(prompts, img, img_labels)
        all_loss = 0
        # auxiliary_loss = self.alpha*FGA["contrastive_loss"] + self.gamma * FGA["orthogonal_loss"] + self.delta * FGA['graph_align_loss']
        if self.weighting_strategy == "uncertain_based_weight":
          classification_loss = Cls["loss_value"] /  torch.square(self.beta)+ torch.log(self.beta)
          class_ortho_loss = FGA["orthogonal_loss"] / torch.square(self.gamma) + torch.log(self.gamma)
          class_clip_loss = FGA["contrastive_loss"]/torch.square(self.alpha) + torch.log(self.alpha) 
          regression_loss = FGA['graph_align_loss'] / (2*torch.square(self.beta))+ torch.log(self.delta)
          # all_loss = self.beta*Cls["loss_value"] + auxiliary_loss
          if FGA["orthogonal_loss"] != 0:
            all_loss = all_loss + class_ortho_loss
          if FGA["contrastive_loss"] != 0:
            all_loss = all_loss + class_clip_loss
          if FGA['graph_align_loss'] != 0:
            all_loss = all_loss + regression_loss
          if all_loss!=0:
            all_loss = all_loss + classification_loss
          else:
            all_loss =  Cls["loss_value"]
        elif self.weighting_strategy == "task_balance":
          # all_loss = self.beta*Cls["loss_value"] + auxiliary_loss
          if FGA["orthogonal_loss"] != 0:
            self.gamma = torch.nn.Parameter(torch.exp(FGA["orthogonal_loss"].detach())) 
            all_loss = all_loss + torch.exp(FGA["orthogonal_loss"].detach()) * FGA["orthogonal_loss"]
          if FGA["contrastive_loss"] != 0:
            self.alpha =  torch.nn.Parameter(torch.exp(FGA["contrastive_loss"].detach()))
            all_loss = all_loss + torch.exp(FGA["contrastive_loss"].detach()) * FGA["contrastive_loss"]
          if FGA['graph_align_loss'] != 0:
            self.delta = torch.nn.Parameter(torch.exp(FGA['graph_align_loss'].detach()))
            all_loss = all_loss + torch.exp(FGA['graph_align_loss'].detach()) * FGA['graph_align_loss']
          if Cls["loss_value"] != 0:
            self.beta = torch.nn.Parameter(torch.exp(Cls["loss_value"] .detach()))
            all_loss = all_loss + torch.exp(Cls["loss_value"] .detach()) * Cls["loss_value"] 
        else:
          auxiliary_loss = self.alpha*FGA["contrastive_loss"] + self.gamma * FGA["orthogonal_loss"] + self.delta * FGA['graph_align_loss']
          all_loss = self.beta * Cls["loss_value"] + auxiliary_loss
        
        # if self.learnable_weight:  # add regularization to advoid gradient exploration
        print(f'alpha = {self.alpha} and the clip loss is {FGA["contrastive_loss"]}')
        print(f'delta = {self.delta} and the graph_align loss is {FGA["graph_align_loss"]}')
        print(f'beta = {self.beta} and the classification loss is {Cls["loss_value"]}')
        print(f'gamma = {self.gamma} and the orthogonal loss is {FGA["orthogonal_loss"]}')
        print(f"the total loss is {all_loss}\n")
        return all_loss
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测概率
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss