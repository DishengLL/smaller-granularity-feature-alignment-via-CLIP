# Smaller granularity feature alignment via CLIP

### General 
**a feature representation task.**

currently using Classification task as an ultility task, aiming to build a general disease detecter via chest x-ray.

**this project trys to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream task.

### Model  
<img src=".\imgs\structure.png" style="zoom:50%;"></img>

### ToDo
- [x] GPU(CUDA) version code 

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix

- [x] biomedCLIP

- [ ] modality gap shift exploration
  
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs

- [ ] think: contrastive learn in image branch feature extractor part

### Ablation
- [ ] visual branch only
  - [ ]  CLIP visual encoder, transformer1, classifier
    - [ ]  using biomedCLIP visual encoder as backbone, 
  - [ ]  custom visual encoder, transformer1, classifier
- [ ] visual branch + text branch
  - [ ] CLIP visual/text encoder, transformer1/2, classifier
  - [ ] CLIP text encoder, **custom visual encoder**, transformer1/2, classifier
- [ ] semantic + domain level alignment

### SOTA solution
##### model and methodology


### progress
- 2023-11-8: 
  - visualization(UMAP), from the 3d plot, the embedings of each disease in the plot does not provide straightforward insight --- there are not huge differences between original(clip, biomedclip) embeddings and the embeddings generated from my othogonal module.
  - using heatmap with simarity matrix data, my embedding indeed push the diseases farther away from each other than the original(clip, biomedclip) ones.
- 2023-11-10
  - debug -- normalization issue before CE loss part
- 2023-11-11
  - train visual-branch-only version --- no contrastive loss between image and text in this version 
  - even though this version get similar performace(acc), the more unstable then the biomed version
  - contrastive loss module improve the generalizability of model (the preliminary and immature insight)