# Fine-grained Feature Alignment via Contrastive Learning


### General 
**a feature representation task.**

currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.

**This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Model  
<img src=".\imgs\structure.png" style="zoom:50%;"></img>

### ToDo
- [x] GPU(CUDA) version code 
- [ ] polish code -- improve the efficiency of model inference
  - [ ] image opening and preprocessing take tremendous of time  --- preprocess image, and store then into numpy or tensor

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix


- [x] biomedCLIP -- this branch works on biomedCLIP pre-trained model (baseline version using original CLIP)


- [ ] modality gap shift exploration
  
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs

- [ ] think: contrastive learn in image branch feature extractor part

### Ablation
- [ ] visual branch only
  - [ ]  CLIP visual encoder, transformer1, classifier
    - [ ]  using biomedCLIP visual encoder as the backbone, 
  - [ ]  custom visual encoder, transformer1, classifier
- [ ] visual branch + text branch
  - [ ] CLIP visual/text encoder, transformer1/2, classifier
  - [ ] CLIP text encoder, **custom visual encoder**, transformer1/2, classifier
- [ ] semantic + domain level alignment

### SOTA solution
##### model and methodology
[CheXclusion: Fairness gaps in deep chest X-ray classifiers](https://arxiv.org/pdf/2003.00827v2.pdf) -- Average AUC = 84.9   

[Towards long-tailed, multi-label disease classification from chest X-ray: Overview of the CXR-LT challenge](https://arxiv.org/pdf/2310.16112v1.pdf)


### progress
- 2023-11-8: 
  - visualization([UMAP](https://zhuanlan.zhihu.com/p/352461768)), from the 3d plot, the embedding of each disease in the plot does not provide straightforward insight --- there are not huge differences between the original(clip, biomedCLIP) embeddings and the embeddings generated from my orthogonal module.
  - using heatmap with similarity matrix data, my embedding indeed pushes the diseases farther away from each other than the original(clip, biomedCLIP) ones.
- 2023-11-10
  - debug -- normalization issue before CE loss part
- 2023-11-11
  - train visual-branch-only version --- no contrastive loss between image and text in this version 
  - even though this version get similar performace(acc), the more unstable then the biomed version
  - contrastive loss module improves the generalizability of the model (the preliminary and immature insight)
- 2023-11-18
  - complete the training of 4 models
     - biomed, biomed_vision_only, CLIP, CLIP_version_only
     - get a phased conclusion
   


### enviroment
```
conda create -n torch_gpu python=3.8
```
