# Fine-grained Feature Alignment via Contrastive Learning

<div align="center" style="position: relative;">
  <img src=".\imgs\chest.png" alt="Chest Image" width="50%" height="auto">
  <p style="position: absolute; bottom: 0; margin: 0;">Generated by GPT-4</p>
</div>


**A feature representation task.**  
currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.

**This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Abstract

Leveraging multimodal data, the Visual Language Model(VLM) demonstrated impressive capability of bridging the
knowledge in multi-modalities. VLMs like CLIP, Flamingo, and DALL-E, which are trained based on the tremendous
amount of data and computational resources show good performance in many different downstream tasks due to the good generalizability. However, like a double-edged sword, the generalizability of pre-trained VLMs limits their performance in the customized setting. In this project, I try to leverage the prior knowledge in pre-trained VLMs and customize the embedding generation in my general classification task. Using the simple contrastive learning method proposed in the report, a robust generalist classifier is available with the deficiency of training data which is a ubiquitous context in the biomedical setting.

> **A feature representation task.**  
> currently using the Classification task as a utility task, aiming to build a general disease detecter via chest x-ray.  
> **This project tries to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream tasks.

### Problem Description:

To alleviate the workload of radiologists, researchers develop algorithms that can automatically classify X-ray images into different classes (corresponding to the existence of different diseases discovered in X-ray images).   
even though, the current SOTA specialist model (customized model for one certain disease) gets very good performance, a generalist model (capable of handling multiple diseases simultaneously) is still weak.  
In this context, what I want to do is establish an algorithm which capable of detecting multiple diseases from X-ray images.  

### Challenges:

To solve this problem, I need to interface 2 main challenges

1. data scarcity
   1.  Private organizations, companies, or hospitals may restrict the publication of medical images due to considerations of business profitability and privacy guidelines.
2. weak performance in multiple disease detection 
   1. while disease-specific detection models reach high accuracy, AI models are less competitive in tackling multiple diseases at the same time, and the performance of disease detection drops dramatically.   

### Rationale for Proposed Solutions

1. First, due to the "data-hungry" nature of AI models, the limitation of training samples may be an important factor leading to poor model performance. 
2. Second, due to the weakness of current medical AI in handling multiple diseases, the "generalist" disease detection system integrates multiple disease-specific models, and this configuration triggers the cumbersomeness of the system. Furthermore,  AI algorithms are trained for one certain downstream task, like the detection of pneumonia, and *would not be able to carry out the complete diagnostic exercise of writing a comprehensive radiology report. This narrow, task-specific approach produces inflexible models, limited to carrying out tasks predefined by the training dataset and
   its labels. In current practice, such models typically cannot adapt to other tasks (or even to different data distributions for the same task) without being retrained on another dataset. Of the more than 500 AI models for clinical medicine that have received approval from the Food and Drug Administration, most have been approved for only 1 or 2 narrow tasks[[8]](https://drive.google.com/file/d/1hLrxBBodiHcl9hT77u54xIfCNnYh0uIb/view?usp=drive_link).*
3. Therefore, the research on the alleviation of data scarcity and improving models' performance in multiple diseases is important and necessary for the development of advanced medical AI.

### Potential trigger of the issues.

1. With the development of AI models, the design of models has become more and more sophisticated. While from multilayer perceptron to deep neural networks and then transformer structure, as the increment of number of parameters the demand for training is increasing too, to grasp the essence from the training samples for specific downstream, and avoid overfitting. Therefore, sufficient and informative training data is the prerequisite for the robustness of medical AI.
2. Acting like a human learner, learning one certain skill is more likely to be successful compared with learning multiple tasks at the same time. That is different tasks could be distractions for each other, rather than beneficial with each other during the training phase. Further reflection, this drawback of multi-tasks may result from the lack of a holistic picture of the relationship of different concepts used in different tasks, and then these concepts blur each other in the embedding space  

### Potential Solutions 

1. leveraging the richness of information in textual description to facilitate feature extraction in the disease detection task from chest X-ray images.
2. using a knowledge graph to inject the prior knowledge of different diseases into my feature extractor. Inducing holistic information of different diseases into the pipeline.

### Model  

<img src=".\imgs\methodology_blank.png" style="zoom:50%;"></img>
Using contrastive learning to align the diseases' representation between textual and visual branches, leveraging the power of LLMs to guide the feature extraction in the visual branch.

### Pathology correlation:

 In reality, the diseases diagnosed from X-ray images are supposed to be correlated with each other to some extent. **Therefore**, the orthogonalizing may not make sense.    
 To inject the prior knowledge of this correlation, I use the graph to represent the hierarchical relation between my 14 labels, and I hope this prior knowledge can guide the model learning.

<div align="center" style="position: relative;">
  <img src="./imgs/graph_convert.png" alt="graph relationship" width="70%" height="auto">
  <p style="position: absolute; bottom: 0; margin: 0;">Hierarchical relation tree of 14 labels</p>
</div>     


### AUC comparison among 14 labels 

<div align="center" style="position: relative;">
  <img src=".\imgs\bio_all_vs_bio_version_2label.png" alt="Chest Image" width="100%" height="auto">
  <p style="position: absolute; bottom: 0; margin: 0;">AUC comparison among 14 labels(config: grpah+Orth+Contrastive)</p>
</div>   

Using the same model and training configuration in the visual branch, comparing the performance concerning the AUC metric between the setting of the visual-only version and the visual-plus-textual version. Shown as a graph, I highlight the difference in performance between these two settings. Using <font color=green>green values</font> to indicate the improvement and <font color=red>red values</font> to indicate degradation if the difference is larger than 0.5%.  

From the figure, most of them are the same, and in some certain diseases like `pleural other` get improvement. 


### ToDo   

- [ ] modality gap shift exploration


### Ablation

- [x] visual branch only
  - [x] CLIP visual encoder, transformer1, classifier
    - [x] using biomedCLIP visual encoder as the backbone, 
  - [ ] custom visual encoder, transformer1, classifier
- [ ] visual branch + text branch
  - [x] CLIP visual/text encoder, transformer1/2, classifier
  - [ ] CLIP text encoder, **custom visual encoder**, transformer1/2, classifier
- [ ] semantic + domain level alignment



### Reference

[1] [Towards long-tailed, multi-label disease classification from chest X-ray: Overview of the CXR-LT challenge](https://arxiv.org/pdf/2310.16112v1.pdf)  
[2] [Explainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis](https://arxiv.org/pdf/2303.13391.pdf)  
[3] [CheXclusion: Fairness gaps in deep chest X-ray classifiers](https://arxiv.org/pdf/2003.00827v2.pdf)    
[4] [A Simple General Approach to Balance Task Difficulty in Multi-Task Learning](https://arxiv.org/pdf/2002.04792.pdf)    
[5] [Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)     
[6] [Attentional Mixtures of Soft Prompt Tuning
for Parameter-efficient Multi-task Knowledge Sharing](https://homes.cs.washington.edu/~akari/papers/attempt_preprint.pdf)      
[7] [A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion](https://arxiv.org/pdf/2303.16378.pdf)    


### progress

- 2023-11-8: 
  - visualization([UMAP](https://zhuanlan.zhihu.com/p/352461768)), from the 3d plot, the embedding of each disease in the plot does not provide straightforward insight --- there are no huge differences between the original(clip, biomedCLIP) embeddings and the embeddings generated from my orthogonal module.
  - using heatmap with similarity matrix data, my embedding indeed pushes the diseases farther away from each other than the original(clip, biomedCLIP) ones.
- 2023-11-10
  - debug -- normalization issue before CE loss part
- 2023-11-11
  - train visual-branch-only version --- no contrastive loss between image and text in this version 
  - even though this version get similar performace(acc), the more unstable then the biomed version
  - contrastive loss module improves the generalizability of the model (the preliminary and immature insight)

- 2023-11-18:
  - tune code, retrain 4 models(biomed + with(out) vision, clip + with(out) vision)


Memo:

```
CheXpert is an open-source rule based tool that is built on NegBio. It proceeds in three stages: (1) extraction, (2) classification, and (3) aggregation. In the extraction stage, all mentions of a label are identified, including alternate spellings, synonyms, and abbreviations (e.g. for pneumothorax, the words "pneumothoraces" and "ptx" would also be captured) [8]. Mentions are then classified as positive, uncertain, or negative using local context. Finally, aggregation is necessary as there may be multiple mentions of a label. Priority is given to positive mentions, followed by uncertain mentions, and lastly negative mentions. If a positive mention exists, then the label is positive. Conversely, if a negative and uncertain mention exist, the label is uncertain. These stages are used to define all labels except "No Finding", which is only positive if all other labels except "Support Devices" are negative or unmentioned. More detail is provided in the CheXpert article [8]. The output of CheXpert was saved to a CSV file with one row per study and one column per finding.
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


### 2024-4-1

#### Assumption:

in the previous experiment, using the disease name as text prompts, we hoped these textual embeddings could facilitate the performance of classification tasks in the visual branch. 

Indeed, adding auxiliary losses can improve the overall performance.  

1. The good news: the performance in all diseases does not drop at a statistical level, and in certain diseases, the classification AUC improves. 
2. Bad news: contrastive loss can not be reduced to around 1.7, which is the bridge between textual and visual branches. Even though the performance in all of the diseases does not drop, the improvements are slight

In my ideal case, visual embeddings should be aligned with textual embeddings. However, the unreduced contrastive loss (1.7) between embeddings with 512 dimensions.

Therefore, regardless of the general disease concepts, I try to induce the diagnostic information (`Positive/Negative` Fracture). 

To simplify the experiment, I initially assumed all of the corresponding textual embeddings should be orthogonal with each other.
Eg: **Positive Fracture** `Ortho` **Negative Edma** 