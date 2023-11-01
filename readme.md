# Smaller granularity feature alignment via CLIP

### General 
**a feature representation task.**

currently using Classification task as an ultility task, aiming to build a general disease detecter via chest x-ray.

**this project trys to explore the data cooperation between multimodal data -- text and image**, and then improve AI downstream task.

### Model  
<img src=".\imgs\structure.png" style="zoom:50%;"></img>

### ToDo
- [ ] evaluation  
  overall -- MSE     
  each disease -- confusion matrix

- [ ] biomedCLIP

- [ ] modality gap shift exploration
  
- [ ] vector output expression loss + contrastive loss between 2 different classifiers' outputs

- [ ] think : contrastive learn in image branch feature extractor part 
