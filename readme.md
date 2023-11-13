# Fine-grained feature alignment via contrastive learning

### General 
**a feature representation task.**

currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.

**This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Model  
<img src=".\imgs\structure.png" style="zoom:50%;"></img>

### ToDo
- [x] GPU(CUDA) version code 

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix

- [x] biomedCLIP -- this branch works on biomedCLIP pre-trained model (baseline version using original CLIP)

- [ ] modality gap shift exploration
  
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs

- [ ] think: contrastive learn in image branch feature extractor part


## phase conclusion 
1. the current average accuracy among 13 diseases is **63.13%**,   
   and the performance of the model **stops improving** in the early training stage. 


## modules
1. CLIP image/text encoder
2. transformer-based models in orthogonal/feature-separated modules
3. MLP-based models for classification task 
