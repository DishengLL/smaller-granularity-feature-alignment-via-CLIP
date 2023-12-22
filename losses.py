import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from typing import List
## Loss optimizes parameters in Text_encoder -- done
class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init()

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
class ImageSuperviseLoss(nn.Module):
    def __init__(self,
        model,
        loss_fn=None,
        ):
        super().__init__()
        self.model = model  # Vis_encoder + Classifier
        self.custom = True if loss_fn == None else False
        if model.mode is None:
            raise NotImplementedError(f"take care model mode setting, if it has not been set, the default is multi-label\nCurrent, the program will raise ERROR!!!")
        self.mode = model.mode
        if loss_fn is None:
            if self.mode in ['multilabel','binary']:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def forward(self,
        img_embedding,
        labels=None,
        **kwargs):

        # compute soft-labels, -1: negative, 0: uncertain, 1: positive
        # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
        if self.custom == True:
            outputs = self.model(img_embedding = img_embedding, labels=labels, return_loss=False)
            loss = self.loss_fn(outputs['logits'], labels)
            outputs['loss_value'] = loss
        else:
            outputs = self.model(img_embedding = img_embedding, labels=labels, return_loss=True)
        return outputs

class LG_CLIP_LOSS(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma = 1, delta = 1, MultiTaskModel=None, learnable_weigh = False):
        super().__init__()
        if learnable_weigh: 
          self.alpha = torch.nn.Parameter(torch.randn(1))
          self.beta = torch.nn.Parameter(torch.randn(1))
          self.gamma = torch.nn.Parameter(torch.randn(1))
          self.delta = torch.nn.Parameter(torch.randn(1))
        else:
          self.alpha = alpha
          self.beta = beta
          self.gamma = gamma
          self.delta = delta
        if MultiTaskModel is None:
            raise ValueError("input MultiTaskModel is None!!!!")
        self.model = MultiTaskModel
        

    def forward(self, 
                img = None,
                prompts:list = None,
                img_labels: List[int] = None):
        if img is None:
            raise ValueError("input image_path is None")
        if prompts is None:
            raise ValueError("input prompts is None")
        if img_labels is None:
            raise ValueError("img_label which will be used in Classifier is None")
        _clip_, Cls, Orth = self.model(prompts, img, img_labels)
        all_loss = self.alpha*_clip_["loss_value"] + self.beta*Cls["loss_value"] + self.gamma * Orth["loss_value"]
        return all_loss
        