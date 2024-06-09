from cgitb import text
from math import tan, tanh
from re import L
from scipy import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
from PIL import Image, ImageFile
from zmq import device
ImageFile.LOAD_TRUNCATED_IMAGES = True
import constants as _constants_
from torch import Tensor
import os
from torchvision import models
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.WARNING)
os.environ['CURL_CA_BUNDLE'] = ''
from pathlib import Path
from typing import Tuple
import math
from utils import TransformerWithLearnableQueries
from utils import Transformer_classifier
import losses
from utils import EDA


import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
num_of_diseases = len(_constants_.CHEXPERT_LABELS)  # Desired number of output embeddings
pwd = os.getcwd()

device = "cuda" if torch.cuda.is_available() else "cpu"
class OrthogonalTextEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.encoder_ly = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_ly, num_layers = 8)

    def forward(self, x:Tensor) -> Tensor:
        '''
        expect x - [batch, sequence, dim]
        output y - [batch, sequence, dim]
        '''
        x = self.encoder(x)
        return x

class SplitVisEncoder(nn.Module):
    def __init__(self, n, d_model=512, nhead = 8, layers = 6, hid_dim=2048, drop = 0.01):
        super(SplitVisEncoder, self).__init__()
        self.n = n
        self.input_splits = nn.Linear(d_model, d_model * n)
        self.fc = nn.Linear(d_model * n, d_model * n)
        self.sequential = nn.Sequential(
            self.input_splits,
            self.fc
        )
        self.encoder_ly = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_ly, num_layers = 8)

    def forward(self, x:Tensor, project=False):
        '''
        expect input shape x - [batch, dim]
        output shape y - [batch, n, dim]
        '''
        x = self.sequential(x) # 1 * n * dim
        x = x.view(x.size(0), self.n, -1)   #.permute(1, 0, 2)  # 调整形状为 (n, batch_size, d_model)

        x = self.encoder(x)

        return x #.cuda()
      
class TextBranch(nn.Module):
    def __init__(self, text_embedding_dim = 512, num_transformer_heads = 8, num_transformer_layers = 6, proj_bias = False, nntype = None):
        super().__init__()
        d_model=512
        if nntype == None:
          self.backbone = "clip"
        else:
          self.backbone = nntype
        if nntype == "biovil-t" or nntype == "cxr-bert-s" :
          d_model = 128
        print(_constants_.BOLD + _constants_.BLUE + "in current Text branch, the text backbone for text embedding is: " \
          + _constants_.RESET + self.backbone)            
       
        # text orthogonal 部分
        self.Orth_transformer = OrthogonalTextEncoder(d_model = d_model)
        
    def forward(self, text_features):
        '''
        文字分支：
        输入: b * [prompt1, prompt2, prompt3, ...]
        mid: b x n x 512
        输出: b x [n vectors corresponding with prompts]
        直接输入处理后的文字embedding tensor, 不需要进行模型的推理
        '''
        output = self.Orth_transformer(text_features) 
        return  output

class ImgBranch(nn.Module):
    def __init__(self, text_embedding_dim = 512, num_transformer_heads = 8, num_transformer_layers = 6, proj_bia = False, 
                 nntype = None, backbone_v:str = None, 
                 trainable_PLM:int = 0, trainable_VisionEncoder = False):
        super().__init__()
        self.projection_head = nn.Linear(512, 512, bias=False)
        nlabel = len(_constants_.CHEXPERT_LABELS)
        d_model=512
        self.device = device
        if backbone_v == "densenet":
          self.backbone_v_model = self.densenet().to(device)
          self.backbone = "densenet"
          print(_constants_.BOLD + _constants_.BLUE + "in current image branch, the vis backbone for vis embedding is: " + _constants_.RESET + backbone_v)    
          self.VisEncoder = SplitVisEncoder(nlabel, d_model = d_model, nhead = 8, layers = 6, hid_dim=2048, drop = 0.01).to(device)
          self.disease_encoder = disease_encoder()
          return
        
        if nntype == None:
            self.backbone = "clip"
        else:
            self.backbone = nntype
        print(_constants_.BOLD + _constants_.BLUE + "in current image branch, the vis backbone for vis embedding is: " + _constants_.RESET + self.backbone)            
        
        if self.backbone in ["biomedCLIP", "biomed", "biomedclip",]:
          import open_clip
          """
          There are 12 attention blocks in Vit module
          """
          self.clip_model, preprocess_train, self.clip_processor = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device = device)
          
          if trainable_VisionEncoder: 
            print(_constants_.RED + "\nvit_base_patch16_224 is trainable!!\n" + _constants_.RESET)
            # in default setting, the pre-trained model is trainable
            for param in self.clip_model.parameters():
              param.requires_grad = True
          else:
            print(_constants_.BLUE + "\nvit_base_patch16_224 is fixed!!\n" + _constants_.RESET)
            for param in self.clip_model.parameters():
              param.requires_grad = False
          
          if not trainable_VisionEncoder  and trainable_PLM > 0:
            # the last n attention blocks are trainable
            # just re-train attention blocks
            if trainable_PLM > 12: # this Vit, the number of attention layer block is 12
              raise RuntimeError("double check the number of attention blocks in this models,\
                in current setting, the number of attention blocks in the model is 12.")
            else:  
              print(f"tune the last {trainable_PLM} attention blocks!!")
              for param in self.clip_model.visual.trunk.blocks[12-trainable_PLM: 12].parameters():
                param.requires_grad = True     
            
            self.clip_model.visual.trunk.norm.weight.requires_grad = True
            self.clip_model.visual.trunk.norm.bias.requires_grad = True
            self.clip_model.visual.head.proj.weight.requires_grad = True
          
          
          # Example usage
          embedding_dim = 768
          output_dim = 512
          num_heads = 8
          num_layers = 6
          hidden_dim = 2048
          num_output_tokens = num_of_diseases + 1 # diseases embedding + CLS

          # Initialize the model
          self.disease_visual_encoder = TransformerWithLearnableQueries(input_dim = embedding_dim, num_heads = num_heads, num_layers = num_layers, hidden_dim = embedding_dim,
                                                                        output_dim = output_dim, num_output_tokens = num_of_diseases)
          self.linear = nn.Linear(in_features=embedding_dim, out_features=512, bias=False)
          self.bn = nn.BatchNorm1d(output_dim*num_of_diseases)
          nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

        elif self.backbone == "cxr-bert-s" or self.backbone == "biovil-t":
          from utils.health_multimodal.image.utils import ImageModelType
          from utils.health_multimodal.image import get_image_inference
          self.image_inference_engine = get_image_inference(ImageModelType.BIOVIL_T)
          self.image_inference_engine.to(device)

          # the dimension of the output of biovil_t is 128
          d_model = 128
          
        elif self.backbone == "clip":
          import clip
          self.clip_model, self.clip_processor  = clip.load("/public_bme/data/lds/model_zoo/ViT-B-32.pt", device=device)
          if trainable_VisionEncoder:
            print(_constants_.RED + "\nViT-B-32.pt is trainable!!\n" + _constants_.RESET)
            for param in self.clip_model.parameters():
              param.requires_grad = True
          else:
            print(_constants_.BLUE + "\nViT-B-32.pt is fixed!!\n" + _constants_.RESET)
            for param in self.clip_model.parameters():
              param.requires_grad = False
        #  in this version, Biomed and CLIP model are been frozen 
        else:
          raise NotImplemented("using custom vis backbone which has not be defined!!!!!")

        self.VisEncoder = SplitVisEncoder(nlabel, d_model = d_model, nhead = 8, layers = 6, hid_dim=2048, drop = 0.01).to(device)

    def get_biomedVit_inter_embeddings(self, model, inputs):
      vit_model = model.visual.trunk  # Access the Vision Transformer part of the model
      
      # Calculate number of patches
      # Pass through the patch embedding layer
      x = vit_model.patch_embed(inputs)
      
      # Append class token
      cls_token = vit_model.cls_token.expand(x.shape[0], -1, -1)
      x = torch.cat((cls_token, x), dim=1)
      
      # Add position embeddings
      x = vit_model.pos_drop(x + vit_model.pos_embed)
      
      # Pass through transformer blocks
      for blk in vit_model.blocks:
          x = blk(x)
      
      # Layer normalization
      embeddings = vit_model.norm(x) # [cls, patch1,... patchN] embeddings
      
      return embeddings

    def densenet(self, n_dim = 512):
        model = models.densenet121(weights=None)
        model_dict = model.state_dict()
        pretrained_state = torch.load("/public_bme/data/lds/model_zoo/densenet/densenet121-a639ec97.pth")
        new_state_dict = OrderedDict()
        for k, v in pretrained_state.items():
            if 'denseblock' in k:
                param = k.split(".")
                k1 = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
                new_state_dict[k1] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, n_dim))  
        return model

    def forward(self, image_input):
        '''
        input: img_path, str
        imd : b X 512
        output: b x n x 512
        '''
        
        if "biomedclip" in self.backbone.lower():
          embeddings = self.get_biomedVit_inter_embeddings(model = self.clip_model, inputs = image_input)
          token_diseases =  self.disease_visual_encoder(embeddings) 
          disease_embeddings = self.linear(token_diseases)
          disease_embeddings = F.normalize(disease_embeddings, p=2, dim=-1)
          batch_size, num_output_tokens, embedding_dim = disease_embeddings.size()
          disease_embeddings = disease_embeddings.view(batch_size, -1)
          normalized_disease_embeddings = self.bn(disease_embeddings)
          normalized_disease_embeddings = normalized_disease_embeddings.view(batch_size, num_output_tokens, embedding_dim)
          return normalized_disease_embeddings
          
        elif "clip" in self.backbone.lower():  ## clip fashion -- biomedclip / clip
          image_features = self.clip_model.encode_image(image_input).float()
        elif self.backbone == "densenet":
          image_features = self.backbone_v_model(image_input).float()
        elif self.backbone == "cxr-bert-s" or self.backbone == "biovil-t":
          # image_input is a tensor for image
          if image_input.dim() == 5:
            image_input = torch.squeeze(image_input, dim=1)
          image_features = self.image_inference_engine.get_projected_global_embedding(image_input)
        else:
          raise ValueError(f"do not support backbone: {self.backbone}.")
        output = self.VisEncoder(image_features)
        return output 

class CustomVisEncoder(nn.Module):
    def __init__(self):
        return 

class LGCLIP(nn.Module):
    '''
    Low Granularity CLIP(LGCLIP) --- contrastive learning between image and text embeddings 
    '''
    def __init__(self,
        vision_branch = ImgBranch,
        checkpoint=None,
        vision_checkpoint=None,
        logit_scale_init_value=0.07,
        nntype = None,
        visual_branch_only = False,
        backbone_v = None, 
        graph_align = "NA",
        no_contrastive = False,
        trainable_PLM = 0,
        trainable_VisionEncoder = False, 
        no_orthogonal = False
        ) -> None:
        super().__init__()
        text_proj_bias = False
        assert vision_branch in [ImgBranch, CustomVisEncoder], 'vision_branch should be one of [ImgBranch]'

        self.vision_model = ImgBranch(nntype = nntype, backbone_v = backbone_v, 
                                      trainable_PLM = trainable_PLM, 
                                      trainable_VisionEncoder = trainable_VisionEncoder)
        if not visual_branch_only:
          self.text_model = TextBranch(nntype = nntype)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint)
        self.nntype = nntype
        self.visual_branch_only = visual_branch_only
        self.graph_align = graph_align
        self.no_contrastive = no_contrastive
        self.no_orthogonal = no_orthogonal

    def from_pretrained(self, input_dir=None):
        '''
        If input_dir is None, download pretrained weight from google cloud and load.
        '''
        import wget
        import zipfile
        pretrained_url = None
        if isinstance(self.vision_model, MedCLIPVisionModel):
            # resnet
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_RESNET
            if input_dir is None:
                input_dir = './pretrained/medclip-resnet'
        elif isinstance(self.vision_model, MedCLIPVisionModelViT):
            # ViT
            pretrained_url = constants.PRETRAINED_URL_MEDCLIP_VIT
            if input_dir is None:
                input_dir = './pretrained/medclip-vit'
        else:
            raise ValueError(f'We only have pretrained weight for MedCLIP-ViT or MedCLIP-ResNet, get {type(self.vision_model)} instead.')

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

            # download url link
            pretrained_url = requests.get(pretrained_url).text
            filename = wget.download(pretrained_url, input_dir)

            # unzip
            zipf = zipfile.ZipFile(filename)
            zipf.extractall(input_dir)
            zipf.close()
            print('\n Download pretrained model from:', pretrained_url)
        
        state_dict = torch.load(os.path.join(input_dir, constants.WEIGHTS_NAME))
        self.load_state_dict(state_dict)
        print('load model weight from:', input_dir)

    def encode_text(self, inputs_text:torch.Tensor):
      # 在此， text embeddings中的数据分布是（mean=0， std=1）
      text_embeds = self.text_model(inputs_text)    # text_feature: backbone generated; text_embedding: processed embeddings
      return text_embeds

    def encode_image(self, img_path=None):
        # image encoder
        nor_diseases_embeddings = self.vision_model(img_path)   #img_feature: backbone generated; vision_ouput: processed embeddings
        return nor_diseases_embeddings

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        dim = img_emb.shape[-1]
        # dim = 1
        reshaped_text_embedding = text_emb.squeeze()
        reshaped_image_embedding = img_emb.squeeze()
        if (len(reshaped_image_embedding.shape) == len(reshaped_text_embedding.shape) == 2):
            reshaped_text_embedding = reshaped_text_embedding.unsqueeze(0)
            reshaped_image_embedding = reshaped_image_embedding.unsqueeze(0)
        sim_matrixes = []
        for i, j in zip(reshaped_text_embedding, reshaped_image_embedding):
          sim_matrixes.append((torch.matmul(i, j.t())/dim) * logit_scale)
        return torch.stack(sim_matrixes, dim = 0)   ## each matrix means text-image sim

    def clip_loss(self, similarities: torch.Tensor) -> torch.Tensor:
        batch = 1
        caption_loss = 0
        image_loss = 0
        if len(similarities.shape) == 3 and similarities.shape[0] != 1:
            batch = similarities.shape[0]
        for i in range(batch):
            similarity = similarities[i]
            caption_loss = caption_loss + self.contrastive_loss(similarity)
            image_loss = image_loss + self.contrastive_loss(similarity.T)
        loss =  (image_loss + image_loss) / batch
        return loss / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / logits.norm(dim=-1, keepdim=True)
        # EDA(logits)
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def forward(self,
            input_text:list,
            img_path=None,
            return_loss=True,
            eval = False,
            graph_align = False,
            **kwargs,
            )->torch.Tensor:
      """
      forward 函数只负责模型推理的输出， 无其他功能
      """
      clip_loss = 0 # "no applicable in visual branch case"
      loss = 0
      graph_align_loss = 0
      orthogonal_loss = 0
      text_embeds = 0
      logits_per_image = 0
      nor_diseases_embeddings = self.encode_image(img_path)
      nor_diseases_embeddings.to(device)
      if eval:  # only need image embeddings for following process
        # return 
        return {'img_embeds' : nor_diseases_embeddings, 'text_embeds' : text_embeds,
          'logits_per_image' : logits_per_image, 'loss_value' : loss, }
      if not self.visual_branch_only:    # text branch included
        text_embeds = self.encode_text(input_text).to(device)
        #similarity matrix img2text [0, 1] in multibatch case: the outer matrix contain several inner matrix text-image
        logits_per_image = self.compute_logits(nor_diseases_embeddings, text_embeds) 

        if return_loss:
            if not self.no_contrastive:
              clip_loss = self.clip_loss(logits_per_image)   ## shape [batch, text_sample, image_sample]
              loss = clip_loss
            if self.graph_align != "NA":
              graph_alignment = Hier_graph_align(logits_per_image)
              if self.graph_align == "binary":
                # using cost to represent correlation between different diseases (ps: in cost matrix, diagonal elements are 0)
                prior_graph_tensor = torch.load(pwd + "/../constants/normalized_cost_matrix.pt")   
                graph_align_loss = graph_alignment.get_loss(prior_graph_tensor)
                loss = clip_loss + graph_align_loss
              else:
                raise NotImplemented()
            if self.no_orthogonal:
              orthogonal_loss = 0
            else:
              # orth_diff = Orthogonal_dif().to(device)
              # orth_diff_cal = orth_diff(text_embeds)
              # orthogonal_loss = orth_diff_cal["loss_value"]
              orth_criterion =  GramOrthogonalLoss()
              orth_loss_cal = orth_criterion(text_embeds)
              orthogonal_loss = orth_loss_cal["loss_value"]
      return {'img_embeds' : nor_diseases_embeddings, 'text_embeds' : text_embeds,
          'logits_per_image' : logits_per_image, 'loss_value' : loss, 
          "graph_align_loss" : graph_align_loss, "contrastive_loss" : clip_loss, 
          "orthogonal_loss" : orthogonal_loss}

class PN_classifier(nn.Module):
    def __init__(self,
        num_class = len(_constants_.CHEXPERT_LABELS),
        input_dim=512,
        mode='multiclass',
        num_cat = 3,
        nntype = "clip_fasion",
        **kwargs) -> None:
        '''args:
        vision_model: the LGCLIP vision branch model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        input number:  the number of input embeddings
        '''
        super().__init__()
        if nntype == "biovil-t" or nntype == "cxr-bert-s":
          input_dim = 128
        self.num_dim = num_class # each dim corresponding with each disease
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        self.num_cat = num_cat
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(num_class*input_dim, num_class*input_dim)
            self.cls = nn.Linear(num_class*input_dim, num_class * num_cat)
            # worse performance in reducing model configuration
            # self.fc = nn.Linear(num_class*input_dim, num_class*input_dim//2)
            # self.fc1 = nn.Linear(num_class*input_dim//2, num_class*input_dim//3)
            # self.cls = nn.Linear(num_class*input_dim//3, num_class * num_cat)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
        img_embeddings,  ## original image
        img_label = None,
        return_loss=True,
        multi_CLS = False,
        **kwargs
        ):
        outputs = defaultdict()
        # take embeddings before the projection head
        num_batch = img_embeddings.shape
        num_batch = num_batch[0]
        # the initial image embedding shape should be [batch size, disease number, embedding dimension]
        img_embeddings = img_embeddings.view(num_batch, -1)
        # after the operation, the image embedding shape should be [batch size, disease number * embedding dimension]
        logits = F.relu(self.fc(img_embeddings))
        logits = F.relu(self.fc(logits))
        # worse performance in reducing model configuration
        # logits = F.relu(self.fc1(logits))
        logits = self.cls(logits)
        outputs['logits'] = logits

        nested_list = img_label
        assert img_label is not None

        if multi_CLS:
            raise NotImplemented("have not implemented")

        if img_label is not None and return_loss:
            if type(img_label[0]) is str:
                nested_list = [json.loads(s) for s in img_label]
            img_label = torch.tensor(np.stack(nested_list), dtype=torch.long).to(device)
            logits = logits.view(-1, self.num_cat)
            
            if self.mode == 'multiclass': img_label = img_label.flatten().long()
            loss = self.loss_fn(logits, img_label)
            outputs['loss_value'] = loss
        return outputs    
      
class Attention_classifier(nn.Module):
    def __init__(self,
        num_labels = len(_constants_.CHEXPERT_LABELS),
        input_dim=512,
        mode='multiclass',
        num_cat = 3,
        nntype = "clip_fasion",
        num_layers = 6, 
        num_heads = 8,
        dropout = 0.1, 
        hidden_dim = 256,
        focal_loss = False,
        **kwargs) -> None:
        '''args:
        num_class: number of classes to predict (the number of diseases)
        input_dim: the embedding dim of input
        mode: multilabel, multiclass, or binary
        num_cat: the number of output categories
        '''
        ## network structure: https://raw.githubusercontent.com/DishengL/ResearchPics/main/classifier_transparenent.png

        super().__init__()
        param_dict = kwargs
        self.focal_loss = focal_loss
        labeling_strategy = param_dict['labeling_strategy'] if "labeling_strategy" in param_dict  else "3_class"
        if labeling_strategy == "S1":
          num_cat = 2  # binary classification --- positive and negative 
        if num_labels > 2:
          self.mode =  "multi_label"
        self.num_cat = num_cat
        if num_labels > 2 and self.num_cat > 2:   # positive, negative and uncertain --- multiple classes 
          self.loss_fn = nn.CrossEntropyLoss()   # input logits
          self.cls = nn.Linear(input_dim, self.num_cat) 
        elif num_labels > 2 and self.num_cat == 2:   # positive and dispositive --- binary class
          self.loss_fn = nn.BCEWithLogitsLoss()   # input logits 
          self.model = Transformer_classifier(input_dim, num_layers, hidden_dim, num_heads, dropout, len(_constants_.CHEXPERT_LABELS))
          if focal_loss:
            print("training with focal loss handling imbalanced dataset!")
            self.Focal_loss_fn =  losses.FocalLoss()
          
        else:
          raise NotImplementedError("error happen in classifier class (Initialization)")


    def forward(self,
        token_embeddings,  ## original image
        img_label = None,
        return_loss=True,
        multilabel = False,
        **kwargs
        ):
        outputs = defaultdict()
        print(f"\nthe shape of token_embeddings is {token_embeddings.shape}\n")
        logits = self.model(token_embeddings)
        outputs['logits'] = logits

        nested_list = img_label
        assert img_label is not None
        
        assert self.mode == "multi_label"

        if img_label is not None and return_loss:
          # if logits.shape != img_label.shape:
          #   batch_size = img_label.shape[0]
          #   logits = logits.view(batch_size, -1)
          # if self.mode in ['multiclass', 'binaryclass']: img_label = img_label.flatten().long()
          if not self.focal_loss:
            loss = self.loss_fn(logits, img_label)
          else:
            loss = self.Focal_loss_fn(logits, img_label)
          outputs['loss_value'] = loss
        return outputs       

class classifier(nn.Module):
    def __init__(self,
        num_labels = len(_constants_.CHEXPERT_LABELS),
        input_dim=512,
        mode='multiclass',
        num_cat = 3,
        nntype = "clip_fasion",
        **kwargs) -> None:
        '''args:
        num_class: number of classes to predict (the number of diseases)
        input_dim: the embedding dim of input
        mode: multilabel, multiclass, or binary
        num_cat: the number of output categories
        '''
        ## network structure: https://raw.githubusercontent.com/DishengL/ResearchPics/main/classifier_transparenent.png

        super().__init__()
        param_dict = kwargs
        labeling_strategy = param_dict['labeling_strategy'] if "labeling_strategy" in param_dict  else "3_class"
        if labeling_strategy == "S1":
          num_cat = 2  # binary classification --- positive and negative 
        if nntype == "biovil-t" or nntype == "cxr-bert-s":
          input_dim = 128
        if num_labels > 2:
          self.mode =  "multi_label"
        self.num_cat = num_cat
        if num_labels > 2 and self.num_cat > 2:   # positive, negative and uncertain --- multiple classes 
            self.loss_fn = nn.CrossEntropyLoss()   # input logits
            self.cls = nn.Linear(input_dim, self.num_cat) 
        elif num_labels > 2 and self.num_cat == 2:   # positive and dispositive --- binary class
            self.loss_fn = nn.BCEWithLogitsLoss()   # input logits 
            self.cls = nn.Linear(num_labels * input_dim, num_labels)
        else:
          raise NotImplementedError("error happen in classifier class (Initialization)")
        self.fc = nn.Linear(num_labels * input_dim, num_labels * input_dim)
        
        

    def forward(self,
        img_embeddings,  ## original image
        img_label = None,
        return_loss=True,
        multilabel = False,
        **kwargs
        ):
        outputs = defaultdict()
        batch_size = img_embeddings.shape[0]
        img_embeddings = img_embeddings.view(batch_size, -1)    
        logits = F.relu(self.fc(img_embeddings))
        logits = F.relu(self.fc(logits))
        logits = self.cls(logits)
        outputs['logits'] = logits

        nested_list = img_label
        assert img_label is not None
        
        assert self.mode == "multi_label"

        if img_label is not None and return_loss:
            if logits.shape != img_label.shape:
              batch_size = img_label.shape[0]
              logits = logits.view(batch_size, -1)
            # if self.mode in ['multiclass', 'binaryclass']: img_label = img_label.flatten().long()
            loss = self.loss_fn(logits, img_label)
            outputs['loss_value'] = loss
        return outputs
    
class Orthogonal_dif(nn.Module):
    '''
    Orthogonal module -- input in predefined text list, pass text branch get n text embeddings. 
    '''
    def __init__(self,
                 logit_scale_init_value = 0.07):
        super().__init__()          
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

    def forward(self,
        text_embeds,
        return_loss=True,
        **kwargs,
        ):
        loss = 0
        _len_ = len(text_embeds.shape)
        multi_logits_per_text = []
        if _len_ == 2:  ## just one sample
          if return_loss:
              logits_per_text =  self.compute_logits(text_embeds)
              loss = self.contrastive_loss(logits_per_text)
          return {'text_embeds':text_embeds,
                  'loss_value':loss, 
                  'multi_logits_per_text':logits_per_text}
        else:   ## multiple samples
          if return_loss:
              print("in the orthogiinal module, the number of text samples is: ", text_embeds.shape[0])
              n_text_sample = (text_embeds.shape[0])
              for sample in text_embeds:
                logits_each_sample = self.compute_logits(sample)
                multi_logits_per_text.append(logits_each_sample)
                loss = loss + self.contrastive_loss(logits_each_sample)
              multi_logits = torch.stack(multi_logits_per_text, dim = 0)
              loss = loss / n_text_sample
          return {'text_embeds':text_embeds,
                  'loss_value':loss, 
                  'multi_logits_per_text':multi_logits}

    def compute_logits(self, emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        # print(f"logit_scale: {logit_scale}")
        logits_per_text = torch.matmul(emb, emb.t()) * logit_scale
        return logits_per_text.t()

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
class Hier_graph_align():
  """
  this class tries to align the prior pathological knowledge with the disease embedding generated
  in the intermediate module 
  
  Funcs:
  1. init
  2. forward
  
  Return:
   the total cosine similarity between diseases and target (scaler)
  """
  def __init__(self, text_image_sim_matrix):
    self.sim_matrix = text_image_sim_matrix
    high_order_disease_corr = []
    for matrix in text_image_sim_matrix:
      high_order = torch.matmul(matrix, matrix.t())
      high_order_disease_corr.append(high_order)
    self.high_order_disease_corr = torch.stack(high_order_disease_corr, dim = 0)  # shape = (batch_size, n_disease, n_disease)
    
  def adjust_datarange(self, tensor, type = "min_max"):
    minv = torch.min(tensor)
    maxv = tensor.max()
    
    if type == "zero2one":
      if (minv < 0 or minv > 1 or maxv < 0 or maxv > 1):
        scaled_vec  = (tensor - minv) / (maxv - minv)
        return scaled_vec
      else:
        return tensor
    if type == "min_max":
      scaled_vec  = (tensor - minv) / (maxv - minv)
      scaled_vec = 2 * scaled_vec - 1
      return scaled_vec
       
  def get_loss(self,
        target_matrix : torch.Tensor,
        data_process_type: str = 'NA'
        ):
    '''
    data_process_type works for data preprocessing  --- target and generated matrix
    data_process_type: the preprocess operation for the target matrix
    '''
    n_disease = target_matrix.shape[0]
    tot_cos_dis = 0
    if (data_process_type == 'softmax'):
      if (target_matrix.sum() != n_disease*n_disease or self.sim_matrix.sum() != n_disease*n_disease):
        logging.info("softmax operation!!")
        target_matrix = torch.nn.functional.softmax(target_matrix, dim=1)
        self.sim_matrix = torch.nn.functional.softmax(self.sim_matrix, dim=1)
    elif (data_process_type == 'normalization'):
      logging.info("normalization operation")
      target_matrix = target_matrix / torch.norm(target_matrix, p=2, dim=1, keepdim=True)
      self.sim_matrix = self.sim_matrix / torch.norm(self.sim_matrix, p=2, dim=1, keepdim=True)
    elif (data_process_type == "NA"):
      logging.info("do not implement data preprocess")
    else:
      raise NotImplementedError("must specify the preprocess operation type!")

    batch_size = self.high_order_disease_corr.shape[0]
    for each_sample in self.high_order_disease_corr:
      each_sample_cos_distance = 0
      for i in range(n_disease):
        disease = self.adjust_datarange(each_sample[i , :].float(),  type = "min_max")
        target = target_matrix[i , :].float().to(device)
        target[i] = 1  # the correlation with itself
        target = self.adjust_datarange(target, type = "min_max")
        cosine_similarity = F.cosine_similarity(disease, target, dim=0)
        each_sample_cos_distance += cosine_similarity
      tot_cos_dis = tot_cos_dis + (1 - (each_sample_cos_distance/n_disease))
    avg_tot_cos_dis = tot_cos_dis / batch_size
    return avg_tot_cos_dis


class GramOrthogonalLoss(nn.Module):
    def __init__(self):
        super(GramOrthogonalLoss, self).__init__()
    
    def forward(self, text_embeds):
      batch_number = (text_embeds.shape[0])
      hidden_dim = (text_embeds.shape[-1])
      batch_loss = 0
      for sample in text_embeds:
        G = sample.t() @ sample
        loss = ((G - torch.diag(torch.diag(G))).pow(2).sum().sqrt())/hidden_dim
        batch_loss += loss
      batch_loss = batch_loss / batch_number
        
      return {'text_embeds':text_embeds,
              'loss_value':batch_loss}  


class MultiTaskModel(nn.Module):
    def __init__(self, nntype = "clip", visual_branch_only = False, backbone_v = None, high_order="NA", 
                 no_orthogonal = False, no_contrastive = False, eval = False, **kwargs):
        super().__init__()
        param_dict = kwargs
        self.uncertain_based_weight = param_dict['weight_strategy'] if "weight_strategy" in param_dict else False
        self.labeling_strategy = param_dict['labeling_strategy'] if "labeling_strategy" in param_dict else False
        self.Alignment_Only = param_dict['Alignment_Only'] if "Alignment_Only" in param_dict else False
        focal_loss = param_dict['focal_loss'] if "focal_loss" in param_dict else False
        if not eval:
          self.trainable_PLM = param_dict['trainable_PLM'] 
        else: 
          self.trainable_PLM = 0
        # S1 -- binary classification
        if  (nntype not in ["clip", "biomedclip", "custom", "cxr-bert-s", "biovil-t"]):
            raise ValueError("currently, only support clip, biomedclip and custom NN")
        if visual_branch_only:
            print(_constants_.CYAN+"current program run in visual branch only version (no contrastive learning between images and text)"+_constants_.RESET)
        if "trainable_VisionEncoder" in param_dict:
          trainable_VisionEncoder = param_dict["trainable_VisionEncoder"]
        else:
          trainable_VisionEncoder =  False
        self.Contrastive_Model = LGCLIP(nntype = nntype, visual_branch_only = visual_branch_only, backbone_v= backbone_v, 
                                        graph_align=high_order, no_contrastive = no_contrastive, no_orthogonal = no_orthogonal, 
                                        trainable_PLM = self.trainable_PLM,
                                        trainable_VisionEncoder = trainable_VisionEncoder).to(device)
        # self.PN_Classifier = PN_classifier(nntype=nntype).to(device)
        if not self.Alignment_Only:
          # self.PN_Classifier = classifier(nntype=nntype, labeling_strategy = self.labeling_strategy,).to(device)
          self.PN_Classifier = Attention_classifier(nntype=nntype, labeling_strategy = self.labeling_strategy, 
                                                    focal_loss = focal_loss).to(device)
        if not visual_branch_only:   ## Orthogonal loss is useless in only visual branch case
          self.Orthogonal_dif = Orthogonal_dif().to(device)
        self.visual_branch_only = visual_branch_only
        self.no_orthogonize = no_orthogonal


    def forward(self,         
                prompts:list,
                img = None,
                img_labels = None,
                eval = False):
        '''
        a: contrastive loss between text and visual branch
          a-1: contrastive loss
          a-2: orthogonal loss
          a-3: graph alignment loss
        b: classification loss 
        c: orthogonal loss
        '''
        assert img is not None
        if eval is False:
          assert img_labels is not None
        multi_task = self.Contrastive_Model(input_text = prompts, img_path = img, eval = eval)
        if self.Alignment_Only : # pre-trained configuration
          classification = {"loss_value": 0}
        else:
          classification = self.PN_Classifier(multi_task["img_embeds"], img_labels)
          # classification = self.PN_Classifier(multi_task, img_labels)
      
        return multi_task, classification


### TODO
# baseline model for multi-label classification
# basic CNN model 
# Resnet
# VGG
# Densent
# Inception

