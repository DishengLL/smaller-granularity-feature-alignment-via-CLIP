# Fine-grained Feature Alignment via Contrastive Learning

<div align="center" style="position: relative;">
  <img src=".\imgs\chest.png" alt="Chest Image" width="50%" height="auto">
  <p style="position: absolute; bottom: 0; margin: 0;">Generated by GPT-4</p>
</div>

### General 
**a feature representation task.**

currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.

**This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Model  
<img src=".\imgs\NN_struct.12.3.png" style="zoom:50%;"></img>

### ToDo

- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix
- [ ] modality gap shift exploration
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs
- [ ] Base line model --- CNN ---Densenet121 $_{[3]}$

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

### Reference

[1] [Towards long-tailed, multi-label disease classification from chest X-ray: Overview of the CXR-LT challenge](https://arxiv.org/pdf/2310.16112v1.pdf)

[2] [Xplainer: From X-Ray Observations to
Explainable Zero-Shot Diagnosis](https://arxiv.org/pdf/2303.13391.pdf)

[3] [CheXclusion: Fairness gaps in deep chest X-ray classifiers](https://arxiv.org/pdf/2003.00827v2.pdf)


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
Structured labels
The mimic-cxr-2.0.0-chexpert.csv.gz and mimic-cxr-2.0.0-negbio.csv.gz files are compressed comma delimited value files. A total of 227,827 studies are assigned a label by CheXpert and NegBio. Eight studies could not be labeled due to a lack of a findings or impression section. The first three columns are:

subject_id - An integer unique for an individual patient
study_id - An integer unique for an individual study (i.e. an individual radiology report with one or more images associated with it)
The remaining columns are labels as presented in the CheXpert article [8]:

Atelectasis
Cardiomegaly
Consolidation
Edema
Enlarged Cardiomediastinum
Fracture
Lung Lesion
Lung Opacity
Pleural Effusion
Pneumonia
Pneumothorax
Pleural Other
Support Devices
No Finding
Note that "No Finding" is the absence of any of the 13 descriptive labels and a check that the text does not mention a specified set of other common findings beyond those covered by the descriptive labels. Thus, it is possible for a study in the CheXpert set to have no labels assigned. For example, study 57,321,224 has the following findings/impression text: "Hyperinflation.  No evidence of acute disease.".   this would be assigned a label of "No Finding", but the use of "hyperinflation" suppresses the labeling of no finding. For details see the CheXpert article [8], and the list of phrases are publicly available in their code repository (phrases/mention/no_finding.txt). There are 2,414 studies which do not have a label assigned by CheXpert. Conversely, all studies present in the provided files have been assigned a label by NegBio.

Each label column contains one of four values: 1.0, -1.0, 0.0, or missing. These labels have the following interpretation:

1.0 - The label was positively mentioned in the associated study, and is present in one or more of the corresponding images
e.g. "A large pleural effusion"
0.0 - The label was negatively mentioned in the associated study, and therefore should not be present in any of the corresponding images
e.g. "No pneumothorax."
-1.0 - The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
Explicit uncertainty: "The cardiac size cannot be evaluated."
Ambiguous language: "The cardiac contours are stable."
Missing (empty element) - No mention of the label was made in the report

for `Support Devices`, this column get 4 kind of values(1, 0, -1, missing), most of them are missing and 1, missing>1>0>>-1
```
