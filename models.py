from math import tan, tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage
import os
from collections import defaultdict
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Orthogonal(nn.Module):
    '''
    Orthogonal module -- input in predefined text list, pass text branch get n text embeddings. 
    '''
    def __init__(self,
                 logit_scale_init_value = 0.07):
        super().__init__(Orthogonal, self)
        self.text_module = TextBranch()           
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))


    def encode_text(self, inputs_text:list):
        inputs_text = inputs_text.cuda()
        text_embeds = self.text_model(inputs_text)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds.to("cpu")
    
    def forward(self,
        input_text:list,
        return_loss=True,
        **kwargs,
        ):
        input_text = input_text.cuda().to("cpu")
        # print(f"\033{type(input_text)}, {len(input_text)}\033[0m")
        text_embeds = self.encode_text(input_text)

        logits_per_text = self.compute_logits(text_embeds, text_embeds) #similarity matrix text2text [0, 1]

        if return_loss:
            loss = self.contrastive_loss(logits_per_text)
        else:
            loss = None

        return {'text_embeds':text_embeds,
                'loss_value':loss, 
                'logits_per_text':logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()


    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class OrthogonalTextEncoder(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # Transformer编码器
        self.encoder_ly = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_ly, num_layers = 8)

    def forward(self, x):
        '''
        expect x - [batch, sequence, dim]
        output y - [batch, sequence, dim]
        '''
        # 通过Transformer编码器
        x = self.encoder(x)
        # # 取每个时间步的输出
        # x = x.permute(1, 0, 2)
        return x

class ImgClassifier(nn.Module):
    '''take LGCLIP model with linear heads for supervised classification on images -- positive/negative/uncertain.
    '''
    def __init__(self,
        img_branch,
        num_class,
        input_dim=512,
        mode='multiclass',
        **kwargs) -> None:
        '''args:
        vision_model: the LGCLIP vision branch model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        input number:  the number of input embeddings
        num_class: corresponding with the number of disease --- the dim of output
        '''
        super(ImgClassifier, self).__init__()
        self.model = img_branch
        self.num_class = num_class
        assert mode.lower() in ['multiclass','multilabel','binary']
        self.mode = mode.lower()
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            self.fc = nn.Linear(input_dim, input_dim)
            self.cls = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.fc = nn.Linear(input_dim, 1)

    def forward(self,
        img_path,  ## original image
        labels=None,
        return_loss=True,
        **kwargs,
        ):

        assert labels is not None
        
        outputs = defaultdict()
        image_embeddings = image_embeddings.cuda().to("cpu")
        # take embeddings before the projection head
        img_embeds = self.model(img_path)
        logits = self.fc(img_embeds)
        logits = self.cls(logits)
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs

# class OverallClassifier(nn.model):
#     '''
#     in ablation section, 
#     '''
#     def __init__(self,
#         img_branch,
#         num_class,
#         input_dim=512,
#         mode='multilabel',
#         **kwargs) -> None:
#         '''args:
#         Visual_Classifer_module: receive one embedding, generate "multi-hot" vector indicate multi-labels
#         input_dim: the embedding dim before the linear output layer
#         mode: multilabel
#         input number:  the number of input embeddings
#         num_class: corresponding with the number of disease --- the dim of output
#         '''
#         super(OverallClassifier, self).__init__()
#         self.model = img_branch
#         self.num_class = num_class
#         assert mode.lower() is 'multilabel'
#         self.mode = mode.lower()
#         tanh = nn.Tanh()
#         if num_class > 2:
#             self.loss_fn = nn.BCEWithLogitsLoss()

#             self.fc = nn.Linear(input_dim, input_dim)
#             self.cls = nn.Linear(input_dim, num_class)

#         else:
#             raise ValueError("the number of class is less than 2, cannot be used in multi-labels problem!")
#             self.loss_fn = nn.BCEWithLogitsLoss()
#             self.fc = nn.Linear(input_dim, 1)

#     def forward(self,
#         img_path,  ## original image
#         labels=None,
#         return_loss=True,
#         **kwargs,
#         ):

#         assert labels is not None
        
#         outputs = defaultdict()
#         image_embeddings = image_embeddings.cuda()#.to("cpu")
#         # take embeddings before the projection head
#         img_embeds = self.model(img_path)
#         logits = self.fc(img_embeds)
#         logits = self.cls(logits)
#         logits = self.tanh(logits)
#         outputs['embedding'] = img_embeds
#         outputs['logits'] = logits
#         if labels is not None and return_loss:
#             labels = labels.cuda().float()
#             loss = self.loss_fn(logits, labels)
#             outputs['loss_value'] = loss
#         return outputs

class SplitVisEncoder(nn.Module):
    def __init__(self, n, d_model=512, nhead = 8, layers = 6, hid_dim=2048, drop = 0.01):
        super(SplitVisEncoder, self).__init__()

        # 分割输入向量为n份
        self.n = n
        self.input_splits = nn.Linear(d_model, d_model * n)
        self.fc = nn.Linear(d_model * n, d_model * n)
        self.sequential = nn.Sequential(
            self.input_splits,
            self.fc
        )
        # Transformer编码器
        self.encoder_ly = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_ly, num_layers = 8)


    def forward(self, x, project=False):
        '''
        expect input shape x - [batch, dim]
        output shape y - [batch, n, dim]
        '''
        x = self.sequential(x) # 1 * n * dim
        x = x.view(x.size(0), self.n, -1)   #.permute(1, 0, 2)  # 调整形状为 (n, batch_size, d_model)

        # 通过Transformer编码器
        x = self.encoder(x)

        # 取每个时间步的输出
        # x = x.permute(1, 0, 2)

        return x.cuda()

class TextBranch(nn.Module):
    def __init__(self, text_embedding_dim = 512, num_transformer_heads = 8, num_transformer_layers = 6, proj_bias = False):
        super().__init__()
        # 初始化 CLIP 预训练模型和处理器
        self.projection_head = nn.Linear(512, 512, bias=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        self.clip_model, self.clip_processor  = clip.load("ViT-B/32", device=device)
        # 冻结 CLIP 部分的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # text orthogonal 部分
        self.transformer = OrthogonalTextEncoder()
        
    def forward(self, text_inputs:list):
        '''
        文字分支：
        输入: b * [prompt1, prompt2, prompt3, ...]
        mid: b x n x 512
        输出: b x [n vectors corresponding with prompts]
        '''
        # print(">>>>>>>>>>>>>>>>>>>>>>\n",text_inputs)
        # 输入经过 CLIP 预训练模型
        # text_inputs = torch.cat([clip.tokenize(f"image of {c}") for c in text_inputs]).to(device)
        text_features = []
        # print(f'\033[31mthe type of text_inputs : {type(text_inputs)}\033[0m')
        with torch.no_grad():
            for text_input in text_inputs:
              text_features.append(self.clip_model.encode_text(clip.tokenize(text_input).cuda()).cuda().float())
        # text-features shape - [batch, num of text, dim]
        text_features = torch.stack(text_features, dim = 0)
        output = self.transformer(text_features)
        return output

class ImgBranch(nn.Module):
    def __init__(self, text_embedding_dim = 512, num_transformer_heads = 8, num_transformer_layers = 6, proj_bia = False):
        super().__init__()
        # 初始化 CLIP 预训练模型和处理器
        self.projection_head = nn.Linear(512, 512, bias=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        self.clip_model, self.clip_processor  = clip.load("ViT-B/32", device=device)
        # 冻结 CLIP 部分的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Transformer 部分
        self.VisEncoder = SplitVisEncoder(13, d_model=512, nhead = 8, layers = 6, hid_dim=2048, drop = 0.01)
        
    def forward(self, image_path):
        '''
        input: img_path
        imd : b X 512
        output: b x n x 512
        '''
        # print(f">>>>>>>>>>>>>>>>>{image_path}")
        images = []
        for image in image_path:

            if "/Users/liu/Desktop/school_academy/ShanghaiTech" in image:
                image = image.replace("/Users/liu/Desktop/school_academy/ShanghaiTech", "D://exchange//ShanghaiTech//")
            images.append(self.clip_processor(Image.open(image).convert("RGB")))

        # plt.subplot(2, 4, (image) + 1)
            # plt.imshow(image)
            # plt.xticks([])
            # plt.yticks([])

        # original_images.append(image)
        # images.append(preprocess(image))
        # texts.append(descriptions[name])

        # image = Image.open(os.path.join(skimage.data_dir, image_path)).convert("RGB")

        # image_input = self.clip_processor(image)
        image_input = torch.tensor(np.stack(images))
        # print("the shape of CLIP iamge output: ", image_input.shape)
        # 输入经过 CLIP 预训练模型
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input).float()

        output = self.VisEncoder(image_features.cuda())
        # if project:
        #     output = self.projection_head(output)
        
        return output
    

class LGCLIP(nn.Module):
    '''
    Low Granularity CLIP(LGCLIP) --- contrastive learning between image and text embeddings 
    '''
    def __init__(self,
        vision_branch = ImgBranch,
        checkpoint=None,
        vision_checkpoint=None,
        logit_scale_init_value=0.07,
        ) -> None:
        super().__init__()
        text_proj_bias = False
        assert vision_branch in [ImgBranch], 'vision_branch should be one of [ImgBranch]'

        self.vision_model = ImgBranch()
        self.text_model = TextBranch()

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint)

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

    def encode_text(self, inputs_text:list):
        # inputs_text = inputs_text.cuda()
        text_embeds = self.text_model(inputs_text)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self, img_path=None):
        # image encoder
        vision_output = self.vision_model(img_path).cuda()
        img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        batch = img_emb.shape[0]
        # print (f"\033[31m the number of batches : f{batch}\033[0m")
        # print(img_emb.shape)
        # print(text_emb.shape)
        reshaped_image_embedding = img_emb.view(batch, -1)
        reshaped_text_embedding =text_emb.view(batch, -1)
        logits_per_text = torch.matmul(reshaped_text_embedding, reshaped_image_embedding.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def forward(self,
            input_text:list,
            img_path=None,
            return_loss=True,
            **kwargs,
            ):
            # input_text = input_text.cuda()/

            img_embeds = self.encode_image(img_path).cuda()
            text_embeds = self.encode_text(input_text).cuda()

            logits_per_image = self.compute_logits(img_embeds, text_embeds) #similarity matrix img2text [0, 1]
            logits_per_text = logits_per_image.t() #similarity matrix text2img


            if return_loss:
                loss = self.clip_loss(logits_per_text)
            else:
                loss = None

            return {'img_embeds':img_embeds, 'text_embeds':text_embeds,
                'logits_per_image':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

class PN_classifier(nn.Module):
    def __init__(self,
        num_class = 13,
        input_dim=512,
        mode='multiclass',
        num_cat = 3,
        **kwargs) -> None:
        '''args:
        vision_model: the LGCLIP vision branch model that encodes input images into embeddings.
        num_class: number of classes to predict
        input_dim: the embedding dim before the linear output layer
        mode: multilabel, multiclass, or binary
        input number:  the number of input embeddings
        '''
        super().__init__()
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
        # img_embeddings = img_embeddings.cuda()
        # take embeddings before the projection head
        num_batch = img_embeddings.shape
        num_batch = num_batch[0]
        img_embeddings = img_embeddings.view(num_batch, -1)
        logits = F.relu(self.fc(img_embeddings))
        logits = F.relu(self.fc(logits))
        logits = self.cls(logits)
        outputs['logits'] = logits

        nested_list = img_label
        assert img_label is not None

        if multi_CLS:
            # self.lose_fn_overall = 
            raise NotImplemented("have not implemented")

        if img_label is not None and return_loss:
            # img_label = img_label.cuda().float()
            # print(f"the shape of logit: {logits.shape}")
            if type(img_label[0]) is str:
                nested_list = [json.loads(s) for s in img_label]
            # print(nested_list)
            img_label = torch.tensor(np.stack(nested_list), dtype=torch.long).cuda()
            # print(f"the shape of image_label: {img_label.shape}")
            logits = logits.view(-1, self.num_cat)
            
            # print(f"output.shape: {output.shape}")
            # print(f"img_label.shape: {img_label.flatten().shape}")
            # print(f"\033[31m {img_embeddings.shape}\033[0m")
            # print(f"\033[31m {img_label.shape}\033[0m")
            
            # if len(img_label.shape) == 1: img_label = img_label.view(-1,1)
            if self.mode == 'multiclass': img_label = img_label.flatten().long()
            # print(logits.shape)
            # print(img_label)
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
        # self.text_module = TextBranch()           
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

    
    def forward(self,
        text_embeds,
        return_loss=True,
        **kwargs,
        ):
        # print("the shape if text_embedding: ",text_embeds.shape)
        loss = 0
        
        batch, num_cls, _ = text_embeds.shape
        multi_logits_per_text = []
        # logits_per_text = self.compute_logits(text_embeds, text_embeds) #similarity matrix text2text [0, 1]

        if return_loss:
            for sample in text_embeds:
              logits_each_sample = self.compute_logits(sample)
              multi_logits_per_text.append(logits_each_sample)
              loss += self.contrastive_loss(logits_each_sample)
            multi_logits = torch.stack(multi_logits_per_text, dim = 0)
        return {'text_embeds':text_embeds,
                'loss_value':loss, 
                'multi_logits_per_text':multi_logits}

    def compute_logits(self, emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(emb, emb.t()) * logit_scale
        return logits_per_text.t()


    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CLIP fashion alignment

        self.Contrastive_Model = LGCLIP()
        # img_embedding classifier
        self.PN_Classifier = PN_classifier()
        # text_embeddings differentiator 
        self.Orthogonal_dif = Orthogonal_dif()

    def forward(self,         
                prompts:list,
                img = None,
                img_labels = None):
        assert img is not None
        assert img_labels is not None
        
        a = self.Contrastive_Model(prompts, img)
        b = self.PN_Classifier(a['img_embeds'], img_labels)
        c = self.Orthogonal_dif(a['text_embeds'])
        return a, b, c
    