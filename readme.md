# Smaller granularity feature alignment via CLIP

### General 
**a feature representation task.**

currently using Classification task as an ultility task, aiming to build a general disease detecter via chest x-ray.

**this project trys to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream task.

### Model  
<img src=".\imgs\structure.png" style="zoom:50%;"></img>

### ToDo
- [x] GPU(CUDA) version code 
- [ ] polish code -- improve efficiency of model inference

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix

- [x] biomedCLIP -- this branch works on biomedCLIP pretrained model (baseline version using original CLIP)

- [ ] modality gap shift exploration
  
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs

- [ ] think : contrastive learn in image branch feature extractor part


## phase conclusion 
1. the current accruacy is **58.13%**,   
   and the performance of model **stops improving** in the early training stage. 


## modules
1. CLIP image/text encoder
2. transformer based models in orthogonal/feature sperated modules
3. MLP based models for classification task 
