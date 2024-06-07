import onnx
import onnxruntime
import torch
import models

# 注意：在onnx转换的过程中，各个forward函数返回的必须是一个tensor，不能是list，tuple，dict等其他的数据类型，
#       否则模型的onnx转换失败

config = {"trainable_PLM":0}
      
model_dump = models.MultiTaskModel(nntype="biomedclip", trainable_PLM=0, labeling_strategy = "S1")

# torch.save(model_dump, "dump_model.pt")


model_dump.eval()
device = "cuda" if torch.cuda.is_available() else "cpu" 
model_dump.to(device)
input_names = ['prompts', "imgs", "img_labels"]
output_names = ['output']
 
prompts = torch.randn(1, 14, 512,requires_grad=False).to(device)
imgs = torch.randn(1, 3, 224, 224,requires_grad=False).to(device)
img_labels = torch.randn(1, 14,requires_grad=False).to(device)

# traced_model = torch.jit.trace(model_dump, (prompts, imgs, img_labels))
torch.onnx.export(model_dump, (prompts, imgs, img_labels), 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')
 
