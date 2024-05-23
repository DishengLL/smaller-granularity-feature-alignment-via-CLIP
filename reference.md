#  Reference
[1]. [Label correlation guided discriminative label feature learning for multi-label chest image classification](https://www.sciencedirect.com/science/article/pii/S0169260724000282?fr=RR-2&ref=pdf_download&rr=874a7424cc7ff1e4)  
[2]. [Domain adaptation via Wasserstein distance and discrepancy metric for chest X-ray image classification](https://www.nature.com/articles/s41598-024-53311-w)  
> `Generalizability`: Moreover, the labeling of medical images is usually expensive and time-consuming, especially for the study of image data from multiple imaging centers with diferent machines and equipment, which leads to migration of image distribution due to the diferences in scanning protocols, shooting parameters and angles, and subject groups.

[3]. [Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification](https://www.nature.com/articles/s41598-019-42294-8)

[4].  [An Empirical Analysis for Zero-Shot Multi-Label Classification on COVID-19 CT Scans and Uncurated Reports](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Dack_An_Empirical_Analysis_for_Zero-Shot_Multi-Label_Classification_on_COVID-19_CT_ICCVW_2023_paper.pdf)


---
`Noisy samples`  
1. [BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification](https://arxiv.org/pdf/2203.01937)
---
`multi-labels correlation`  
[1].[Label Semantic Improvement with Graph Convolutional Networks for Multi-Label Chest X-Ray Image Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10505562)
> 使用图卷积网络来graps不同label之间的correlation，旨在提升多标签分类的准确率。 本质是representation learning

[2]. [Advanced Augmentation and Ensemble Approaches for Classifying Long-Tailed Multi-Label Chest X-Rays](https:/openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Nguyen-Mau_Advanced_Augmentation_and_Ensemble_Approaches_for_Classifying_Long-Tailed_Multi-Label_Chest_ICCVW_2023_paper.pdf)
> When diagnosing chest X-rays (CXRs), the problem becomes multi-label as patients often show multiple disease findings simultaneously. Intriguingly, only a few studies have included label co-occurrence knowledge in the learning process [5, 4]. Considering the long-tailed class distribution in chest X-ray datasets, incorporating label cooccurrence information may offer valuable insights for addressing imbalanced and infrequent disease categories in this complex medical imaging task.

[3].
---
`FOUNDATION MODEL`  
[1]. [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4?fromPaywallRec=false)

---   
`Evaluation metric`    
[1]. [A visual way to think of macro and micro averages in classification metrics](https://medium.com/@ehudkr/a-visual-way-to-think-on-macro-and-micro-averages-in-classification-metrics-190285dc927f)    
>  Macro-averaging gives equal weight to each class, while micro-averaging gives equal weight to each instance
 
[2]. [Micro and Macro Averaging](https://sklearn-evaluation.ploomber.io/en/latest/classification/micro_macro.html)
> 1. If the dataset is balanced, both micro-average and macro-average will result in similar scores.
> 2. If the larger class in an imbalanced dataset performs better than the minority classes (the number of True Positives is significantly higher than the number of False Positives), the micro-average score will be higher than the macro-average score.
> 3. In a noisy dataset, the number of True Positives might be significantly lower than the number of False Positives for the majority class. In such a case, the macro-average score will be higher than the micro-average score. But this would be a bit misleading since a large number of examples are not properly classified.

---
`NLP`
[NLP-Powered Insights: A Comparative Analysis for Multi-Labeling Classification with MIMIC-CXR Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10529271)
> 使用report做14label classification

