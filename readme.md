# Fine-grained Feature Alignment via Contrastive Learning

<div style="text-align: center;">
  <img src=".\imgs\chest.png" style="width: 50%; height: auto; margin: 0 auto;">
</div>


### General 
**a feature representation task.**

currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.

**This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Model  
<img src=".\imgs\NN_struct.12.3.png" style="zoom:50%;"></img>

### ToDo
- [x] GPU(CUDA) version code 
- [ ] polish code -- improve the efficiency of model inference
  - [ ] image opening and preprocessing take tremendous of time  --- preprocess image, and store then into numpy or tensor

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix
- [ ] get and store prompt embeddings for reusing 


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

- 2023-11-11-18:
  - tune code, retrain 4 models(biomed + with(out) vision, clip + with(out) vision)


Memo:
```
Each label column contains one of four values: 1.0, -1.0, 0.0, or missing. These labels have the following interpretation:

1.0 - The label was positively mentioned in the associated study, and is present in one or more of the corresponding images
e.g. "A large pleural effusion"
0.0 - The label was negatively mentioned in the associated study, and therefore should not be present in any of the corresponding images
e.g. "No pneumothorax."
-1.0 - The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
Explicit uncertainty: "The cardiac size cannot be evaluated."
Ambiguous language: "The cardiac contours are stable."
Missing (empty element) - No mention of the label was made in the report


for `no finding`, this column gets 3 kind of values(1, missing)

for `Support Devices`, this column get 4 kind of values(1, 0, -1, missing), most of them are missing and 1, missing>1>0>>-1
```
