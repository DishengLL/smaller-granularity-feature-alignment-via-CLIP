# from pathlib import Path
# from typing import Tuple
# import torch
# from health_multimodal.image.utils import ImageModelType
# from health_multimodal.image import get_image_inference
# import logging
# from tqdm import tqdm
# import pandas as pd

# image_path = "/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/imgs/AUCcomp.png"


# class VLM_infer:
#   def __init__(self, image_tensor_path):
#     # logging.DEBUG("initializing VLM infer !")
#     self.image_tensor_path = image_tensor_path
#     return 
  
#   @torch.no_grad()
#   def infer_Biovil_t(self, image_tensor) -> torch.Tensor:
    
#     # """Compute global image embedding in the joint latent space.

#     #     :param image_path: Path to image tensor which has been transformed.
#     #     :return: Torch tensor containing l2-normalised global image embedding [joint_feature_dim,]
#     #              where joint_feature_dim is the dimensionality of the joint latent space.
                 
#     #     reference : https://hi-ml.readthedocs.io/en/latest/api/health_multimodal.image.inference_engine.html
#     #     """
#     #     input_image, _ = self.load_and_transform_input_image(image_path, self.transform)
#     #     projected_img_emb = self.model.forward(input_image).projected_global_embedding
#     #     projected_img_emb = F.normalize(projected_img_emb, dim=-1)

#     #     assert projected_img_emb.shape[0] == 1
#     #     assert projected_img_emb.ndim == 2

#     #     return projected_img_emb[0]
#     image_inference = get_image_inference(ImageModelType.BIOVIL_T)
#     embedding = image_inference.get_projected_global_embedding(image_tensor)
#     return embedding # shape -- n x 128 
  
#   def image_tranfer_Biovil_t(img_path:str = None) -> torch.Tensor:
#     return  

#   def infer_cxr_bert_g(image_path):
#     return 


#   def infer_cxr_bert_s(image_path):
#     return 


# class generate_img_tensor_for_biovil_t:
#   def __init__(self):
#     self.image_inference = get_image_inference(ImageModelType.BIOVIL_T)
#     self.count = 1
#     return 
  
  
#   def process_row2generate_img_tensor_biovil(self, row):
#       # 在这个示例中，我们将列 'a' 中的值加上 10，然后保存到新的列 'B' 中
#       raw_img = row['file_path']

#       if "D:/project_x_ray_CLIP/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files" in raw_img:
#         raw_img = raw_img.replace("D:/project_x_ray_CLIP/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/", "/public_bme/data/lds/")   
#       input_image_tensor, _ = self.image_inference.load_and_transform_input_image(Path(raw_img), self.image_inference.transform)
#       generate_embedding_name = raw_img.split(".")[0] + "_biovil_t.pth"
#       # print(generate_embedding_name)
#       # torch.save(input_image_tensor, generate_embedding_name)
#       return generate_embedding_name
      
    
# def main():
#   ge_embedding = generate_img_tensor_for_biovil_t()
#   train = pd.read_csv("/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/P10_12_train_12_16_labels14.csv", index_col=False)
#   test = pd.read_csv("/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/P10_12_test_12_16_labels14.csv", index_col=False)
  
#   train["biovil_t_img_tensor_path"]= train.apply(ge_embedding.process_row2generate_img_tensor_biovil, axis=1)
#   test["biovil_t_img_tensor_path"].apply(ge_embedding, axis=1)
#   train.to_csv('/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/mimic-cxr-train/P10_12_train_1_29_labels14_biovil_t.csv', index=False)  # 如果不想保存索引，设置 index=False
#   print(">>>>> complete training data")
#   del train
#   test["biovil_t_img_tensor_path"].apply(ge_embedding.process_row2generate_img_tensor_biovil, axis=1)
#   test.to_csv('/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/mimic-cxr-train/P10_12_test_1_29_labels14_biovil_t.csv', index=False)  # 如果不想保存索引，设置 index=False
# # basic: bert: torch.Size([14, 768])
# # basic: biomedclip: torch.Size([14, 512])
# # basic: biovil_t: torch.Size([14, 128])
# # basic: clip: torch.Size([14, 512])
# # basic: cxr_bert_g: torch.Size([14, 768])
# # basic: cxr_bert_s: torch.Size([14, 128])
# # detailed: bert: torch.Size([14, 768])
# # detailed: biomedclip: torch.Size([14, 512])
# # detailed: biovil_t: torch.Size([14, 128])
# # detailed: clip: torch.Size([14, 512])
# # detailed: cxr_bert_g: torch.Size([14, 768])
# # detailed: cxr_bert_s: torch.Size([14, 128])
# # a = VLM_infer(image_path)
# # img = torch.tensor([1,2])
# # a.infer_Biovil_t( Path("/home_data/home/v-liudsh/coding/plot.png"))
# # print(type(a.image_inference.model))
# # print(a.image_inference.resize_size, a.image_inference.crop_size)
# # print(a.image_inference.to)

# if __name__ == "__main__":
#   main()
from pathlib import Path
from typing import Tuple
import torch
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image import get_image_inference
import logging
from tqdm import tqdm
import pandas as pd

class generate_img_tensor_for_biovil_t:
  def __init__(self):
    self.image_inference = get_image_inference(ImageModelType.BIOVIL_T)
    self.count = 1
    return 
  
  
  def process_row2generate_img_tensor_biovil(self, image, dest):
      # 在这个示例中，我们将列 'a' 中的值加上 10，然后保存到新的列 'B' 中

      if "D:/project_x_ray_CLIP/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files" in image:
        image = image.replace("D:/project_x_ray_CLIP/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/", "/public_bme/data/lds/")   
      input_image_tensor, _ = self.image_inference.load_and_transform_input_image(Path(image), self.image_inference.transform)
      torch.save(input_image_tensor, dest)
      return input_image_tensor
      


training = pd.read_csv('/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/P10_12_train_1_29_labels14_biovil.csv', index_col=0)
print(training.iloc[0])

img_paths = training.file_path
tensor_path = training.Biovil_img_tensor_path
total = len(tensor_path)
print(total)
print(len(tensor_path), len(img_paths))
dev = total // 10
count = 0
# for i, j in enumerate(img_paths):
#   print(type(j), j,  tensor_path[i])
#   CLIP_Process(str(j), tensor_path[i])
#   if i % dev == 0:
#     print(i)
#   else:
#     continue
biovil = generate_img_tensor_for_biovil_t()
print("\n---train begin---- \n")
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
for  (img_path, tensor_path) in (zip(img_paths, tensor_path)):
  try:
    # print( type(img_path), img_path, type(ten,,l[pl-0o-or_path), tensor_path)
    biovil.process_row2generate_img_tensor_biovil(img_path, tensor_path)
    if count%dev == 0:
      print(count/dev)
      print(img_path, tensor_path)
    count+=1
  except Exception as e:
    print(e)
    
del training 
print("\n---train end---- \n")
print("\n---------- \n")
print("\n----test begin--- \n")
testing = pd.read_csv("/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/data/mimic-cxr-train/P10_12_test_1_29_labels14_biovil.csv", index_col=0)
img_paths = testing.file_path
tensor_path = testing.Biovil_img_tensor_path
total = len(tensor_path)
print(total)
print(len(tensor_path), len(img_paths))
dev = total // 10
count = 0
# for i, j in enumerate(img_paths):
#   print(type(j), j,  tensor_path[i])
#   CLIP_Process(str(j), tensor_path[i])
#   if i % dev == 0:
#     print(i)
#   else:
#     continue
biovil = generate_img_tensor_for_biovil_t()

for  (img_path, tensor_path) in (zip(img_paths, tensor_path)):
  try:
    # print( type(img_path), img_path, type(ten,,l[pl-0o-or_path), tensor_path)
    biovil.process_row2generate_img_tensor_biovil(img_path, tensor_path)
    if count%dev == 0:
      print(count/dev)
      print(img_path, tensor_path)
    count+=1
  except Exception as e:
    print(e)
