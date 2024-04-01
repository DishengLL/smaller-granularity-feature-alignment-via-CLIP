import os
import constants as C
import torch 
from PIL import Image
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from scipy.stats import norm


device = "cuda" if torch.cuda.is_available() else "cpu"
class PLM_embedding:
  '''
  draw heatmap based on the cosine sim. between embeddings generated by Pre-trained Language Model (PLM)
  '''
  def __init__(self):
    self.context_length = 256
    return
    
  def get_clip_text_embeddings(self, text:list):
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    del text
    return text_features
  
  def get_biomedclip_text_embedding(self, text:list):
    from open_clip import create_model_from_pretrained, get_tokenizer 
    image = "./imgs/chest.png"
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device)
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    texts = tokenizer(text, context_length=self.context_length).to(device)
    dummy_img = torch.stack([preprocess(Image.open(image))]).to(device)
    with torch.no_grad():
        _, text_features, _ = model(dummy_img, texts)
    return text_features
  
  def get_biovil_text_embedding(self, text_prompts:list):
    import torch
    from transformers import AutoModel, AutoTokenizer
    # Load the model and tokenizer
    url = "microsoft/BiomedVLP-BioViL-T"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    model = AutoModel.from_pretrained(url, trust_remote_code=True)

    with torch.no_grad():
        tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                                      add_special_tokens=True,
                                                      padding='longest',
                                                      return_tensors='pt')
        embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                    attention_mask=tokenizer_output.attention_mask)

        # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
        # sim = torch.mm(embeddings, embeddings.t())
        return embeddings
  
  def get_biovil_CXR_BERT_specialized_text_embedding(self, text_prompts:list):
    url = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    model = AutoModel.from_pretrained(url, trust_remote_code=True)

    # Tokenize and compute the sentence embeddings
    with torch.no_grad():
      tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                                  add_special_tokens=True,
                                                  padding='longest',
                                                  return_tensors='pt')
      embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                    attention_mask=tokenizer_output.attention_mask)
    return embeddings
  
  def get_biovil_CXR_BERT_general_text_embedding(self, text_prompts:list):
    '''
    First, we pretrain CXR-BERT-general from a randomly initialized BERT model via Masked Language Modeling (MLM) on abstracts 
    PubMed and clinical notes from the publicly-available MIMIC-III and MIMIC-CXR. 
    In that regard, the general model is expected be applicable for research in clinical domains other than the chest radiology 
    through domain specific fine-tuning.
    
    using average embedding in the sequence to represent sentence embedding
    '''
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    with torch.no_grad():
      encoded_input = tokenizer(text_prompts, return_tensors='pt', padding=True, )
      output = model(**encoded_input, output_hidden_states=True)
    del encoded_input
    del model
    del tokenizer
    avg_embedding = torch.mean(output.hidden_states[-1], axis=1)
    del output
    return avg_embedding
  
  def get_bert_text_embedding(self, text_prompts:list):
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    with torch.no_grad():
      encoded_input = tokenizer(text_prompts, return_tensors='pt', padding=True)
      output = model(**encoded_input)
    cls_embedding = (output.pooler_output)
    '''
    Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining
    '''
    del encoded_input
    del output
    return cls_embedding

    
class plot():
  def __init__(self):
    return
  
  def plot_heat_map(self, embedding_matrix:Tensor, prompt:list, *args, **kwargs):
    if device == "cuda":
      embedding_matrix = embedding_matrix.to('cpu')
    embedding_matrix_np = embedding_matrix.numpy()
    cosine_similarities = cosine_similarity(embedding_matrix_np)
    template = ""
    backbone = ""
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    if "template" in kwargs:
      template = kwargs["template"]
    if "backbone" in kwargs:
      backbone = kwargs['backbone']    
        
    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1.2)  
    sns.set(font_scale=1.2)  # 调整字体大小
    sns.set_style("whitegrid")  # 设置白色背景和网格线
    vmin = 0
    if np.min(cosine_similarities) < 0:
      vmin = np.min(cosine_similarities)
    heatmap = sns.heatmap(cosine_similarities, annot=True, cmap="RdYlBu_r", fmt=".2f",
                          xticklabels=prompt,
                          yticklabels=prompt, vmin=vmin, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha="right")
    heatmap.set_xlabel("text1", fontsize=14)
    heatmap.set_ylabel("text2", fontsize=14)
    title_text = plt.title(f"cos sim in {backbone} using {template} template", fontsize=16)
    title_text.set_fontweight("bold")
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f".\output\sim_heatmap\{backbone}_{template}.png")
    plt.show()
    return 

  def plot_sim_distribution(self, embedding_matrix:Tensor, prompt:list, *args, **kwargs):
    return 
  
  def plot_all(self, embedding_matrix:Tensor, prompt:list, *args, **kwargs, ):
    fig, (ax1, ax2, ) = plt.subplots(1, 2, figsize=(32, 12))
    ax2.grid()
    if device == "cuda":
      embedding_matrix = embedding_matrix.to('cpu')
    embedding_matrix_np = embedding_matrix.numpy()
    cosine_similarities = cosine_similarity(embedding_matrix_np)
    template = ""
    backbone = ""
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    if "template" in kwargs:
      template = kwargs["template"]
    if "backbone" in kwargs:
      backbone = kwargs['backbone']    
  
    sns.set(font_scale=0.7)  # 调整字体大小
    sns.set_style("whitegrid")  # 设置白色背景和网格线
    vmin = 0
    if np.min(cosine_similarities) < 0:
      vmin = np.min(cosine_similarities)
    heatmap = sns.heatmap(cosine_similarities, annot=True, cmap="RdYlBu_r", fmt=".2f",
                          xticklabels=prompt,
                          yticklabels=prompt, vmin=vmin, vmax=1, ax = ax1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha="right")
    heatmap.set_xlabel("text1", fontsize=14)
    heatmap.set_ylabel("text2", fontsize=14)
    title_text = plt.title(f"cos sim in {backbone} using {template} template", fontsize=16)
    title_text.set_fontweight("bold")
    
    sns.kdeplot(cosine_similarities.flatten(), fill=True, ax=ax2, alpha=0.7)
    mean_value, std_dev = norm.fit(cosine_similarities.flatten())
    ax2.set_title('Density Plot')
    ax2.set_xlabel('Similarity Values')
    ax2.set_ylabel('Density')
    ax2.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    ax2.axvline(mean_value + std_dev, color='green', linestyle='dashed', linewidth=2, label=f'Std: {std_dev:.2f}')
    ax2.axvline(mean_value - std_dev, color='green', linestyle='dashed', linewidth=2)



    plt.tight_layout()  # 自动调整布局
    plt.legend()
    # plt.show(plt.savefig(f".\output\sim_heatmap\{backbone}_{template}.png"))
    if not os.path.exists('./output/sim_heatmap/'):
        os.makedirs('./output/sim_heatmap/')
    plt.savefig(f"./output/sim_heatmap/{backbone}_{template}_all.png")
    print(f"saving in .\output\sim_heatmap\{backbone}_{template}_all.png")
    return 
    
    
class Prompt():
  def __init__(self):
    return 
  
  def compose_details():
    return
  
  def compose_cls():
    return 
  
  def get_prompts(self, template:str)->list:
    if template == "basic":
      return C.BASIC_PROMPT
    elif template == "detailed":
      return list(C.DESC_PROMPT.values())
    elif template == "diagnostic":
      return C.DIAGNOSTIC_CHEXPERT_LABELS
    elif template == "classification":
      return 
      

def main():
    parser = argparse.ArgumentParser(description="get text embedding and plot heatmap")
    parser.add_argument("--backbone", '-b',  type=str, help="specify the embedding model.", choices = ["clip", 'biomed', "biovil_t", "CXR_BERT_s","CXR_BERT_g", "bert"], required=True)
    parser.add_argument("--template", '-t',  type=str, default='basic', choices = ["detailed", "basic", "diagnostic"],help="specify the prompt template(default: basic).")
    args = parser.parse_args()
    backbone = args.backbone
    template = args.template
    prompt = Prompt()
    prompt = prompt.get_prompts(template)
    PLM = PLM_embedding()
    if backbone == "clip":
      text_embedding = PLM.get_clip_text_embeddings(prompt)
    # print(clip_text_embedding.shape)
    elif backbone == "biomed":
      text_embedding = PLM.get_biomedclip_text_embedding(prompt)
    elif backbone == "biovil_t":
      text_embedding = PLM.get_biovil_text_embedding(prompt)
    # assert clip_text_embedding.shape == biomedclip_text_embedding.shape == biovil_text_embedding.shape
    elif backbone == "CXR_BERT_g":
      text_embedding = PLM.get_biovil_CXR_BERT_general_text_embedding(prompt)
    elif backbone == "CXR_BERT_s":
      text_embedding = PLM.get_biovil_CXR_BERT_specialized_text_embedding(prompt)
    elif backbone == "bert":
      text_embedding = PLM.get_bert_text_embedding(prompt)
    else:
      raise NoImplement("have not been implemented")
    
    # print((text_embedding.hidden_states))
    # for i in text_embedding:
    #   print(i)
    # print(type(text_embedding.logits))
    # print((text_embedding.logits).shape)  
    painter = plot()
    # painter.plot_heat_map(text_embedding, C.CHEXPERT_LABELS, template = template, backbone=backbone)
    painter.plot_all(text_embedding, prompt, template = template, backbone=backbone)
if __name__ == "__main__":
  main()
  print(C.GREEN + "complete!" + C.RESET)
