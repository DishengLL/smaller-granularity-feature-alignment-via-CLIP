import pdb
import os
import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import constants 
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torchmetrics


def calculate_auc(pred, target):
  assert pred.shape[-1] == len(constants.CHEXPERT_LABELS)
  if pred.device.type == "cuda":
    pred = pred.cpu()
  if target.device.type == "cuda":
    target = target.cpu()
  each_auc = {}
  # metric = {
  #   'micro/auc': roc_auc_score(y_true=target, y_score=pred, average='micro'),
  #   'macro/auc': roc_auc_score(y_true=target, y_score=pred, average='macro'),
  try:
      metric = {
          'macro/auc': roc_auc_score(y_true=target, y_score=pred, average='macro'),
          'micro/auc': roc_auc_score(y_true=target, y_score=pred, average='micro')
      }
  except ValueError as e:
      print(f"Error calculating AUC: {e}")
      metric =  {'macro/auc': None, 'micro/auc': None}
  for disease in range(len(constants.CHEXPERT_LABELS)):
    disease_pred = pred[:, disease]
    disease_target = target[:, disease]
    unique_classes = torch.unique(disease_target)
    if len(unique_classes) < 2:
      print("Warning: Only one class present. AUC score cannot be calculated.")
      auc = -1
    else:
      auc = roc_auc_score(y_true = disease_target, y_score = disease_pred)
    each_auc[constants.CHEXPERT_LABELS[disease]] = auc
  metric["disease_auc"] = each_auc
  return metric
    
class Evaluator:
    '''do evaluation on chexpert5x200 zero-shot classification
    '''
    def __init__(self,
        FG_model_cls,
        eval_dataloader=None,
        labeling_strategy = None, 
        dataset = None,
        ) -> None:
        '''specify class_names if doing zero-shot classification.
        mode: `binary`, 'multiclass`, or `multilabel`,
        if set None, the method will automatically decide from data.
        recommend to set explicitly to avoid errors.
        '''
        self.clf = FG_model_cls
        self.eval_dataloader = eval_dataloader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labeling_strategy = labeling_strategy
        self.mode = "binary" if labeling_strategy in ["S1"] else "multiclass"
        if dataset:
          self.dataset = dataset 
        else:
          raise RuntimeError("error: dataset is required.")
        self.dataset_thresholds = {
          "MIMIC": constants.MIMIC_CXR_thresholds,
          "CheXpert": constants.CheXpert_thresholds,
          "NIH": constants.NIH_thresholds,
        }
        self.thresholds = torch.tensor(list(self.dataset_thresholds[self.dataset].values())).to(self.device)
        self.labels = list(self.dataset_thresholds[self.dataset].keys())
        num_class = len(self.labels )
        
        self.precision = torchmetrics.Precision(task = "multilabel", num_labels=num_class, average='macro').to(self.device)
        self.recall = torchmetrics.Recall(task = "multilabel", num_labels=num_class, average='macro').to(self.device)
        self.f1_score = torchmetrics.F1Score(task = "multilabel",num_labels=num_class, average='macro').to(self.device)
        self.precision_micro = torchmetrics.Precision(task = "multilabel", num_labels=num_class, average='micro').to(self.device)
        self.recall_micro = torchmetrics.Recall(task = "multilabel", num_labels=num_class, average='micro').to(self.device)
        self.f1_score_micro = torchmetrics.F1Score(task = "multilabel",num_labels=num_class, average='micro').to(self.device)
        self.precision_weighted = torchmetrics.Precision(task = "multilabel", num_labels=num_class, average='weighted').to(self.device)
        self.recall_weighted = torchmetrics.Recall(task = "multilabel", num_labels=num_class, average='weighted').to(self.device)
        self.f1_score_weighted = torchmetrics.F1Score(task = "multilabel",num_labels=num_class, average='weighted').to(self.device)

    def calculate_metrics(self, predict_proba, target):
        pred = (predict_proba > self.thresholds).to(torch.float32)
        pred = pred.to(self.device)
        metrics = self.get_P_R_F1(predictions = pred, true_labels = target)
        
        return metrics
      
      
    def get_P_R_F1(self, predictions, true_labels):
      '''
      传入的predictions是binary predicted labels
      '''
      # For multi-class classification

      # 更新指标，传递概率矩阵和真实标签
      true_labels = true_labels.to(self.device)
      self.precision.update(predictions, true_labels)
      self.recall.update(predictions, true_labels)
      self.f1_score.update(predictions, true_labels)
      self.precision_micro.update(predictions, true_labels)
      self.recall_micro.update(predictions, true_labels)
      self.f1_score_micro.update(predictions, true_labels)
      self.precision_weighted.update(predictions, true_labels)
      self.recall_weighted.update(predictions, true_labels)
      self.f1_score_weighted.update(predictions, true_labels)
      # 计算结果
      precision_result = self.precision.compute()
      recall_result = self.recall.compute()
      f1_score_result = self.f1_score.compute()
      precision_result_micro = self.precision_micro.compute()
      recall_result_micro = self.recall_micro.compute()
      f1_score_result_micro = self.f1_score_micro.compute()
      precision_result_weighted = self.precision_weighted.compute()
      recall_result_weighted = self.recall_weighted.compute()
      f1_score_result_weighted = self.f1_score_weighted.compute()

      return {
        'micro/precision':precision_result_micro,
        'micro/recall': recall_result_micro,
        'micro/f1': f1_score_result_micro,
        'macro/precision': precision_result,
        'macro/recall': recall_result,
        'macro/f1': f1_score_result,
        'weighted/precision': precision_result_weighted,
        'weighted/recall': recall_result_weighted,
        'weighted/f1': f1_score_result_weighted,
        } 

    # TODO: add more metrics, and codding
    def get_average_precision_score(self, pred_probs, labels):
      true_labels = labels.cpu().numpy()
      pred_probs = pred_probs.cpu().numpy()
      # 计算每个标签的 average precision score
      average_precision_per_label = average_precision_score(true_labels, pred_probs, average=None)
      average_precision_per_label_micro = average_precision_score(true_labels, pred_probs, average="micro")
      average_precision_per_label_weighted = average_precision_score(true_labels, pred_probs, average="weighted")
      average_precision_mean = np.mean(average_precision_per_label)
      average_precision_mean_micro = np.mean(average_precision_per_label_micro)
      average_precision_mean_weighted = np.mean(average_precision_per_label_weighted)
      return (average_precision_per_label, average_precision_mean, average_precision_per_label_micro, average_precision_mean_micro, average_precision_per_label_weighted, average_precision_mean_weighted)

    def get_AUPRC(self, pred_probs, labels):
      y_true = labels.cpu().numpy()
      y_score = pred_probs.cpu().numpy()
      
      num_labels = y_true.shape[-1]  # Number of labels
      individual_auprc = []

      for label in range(num_labels):
          y_true_label = y_true[:, label]
          y_score_label = y_score[:, label]
          if np.isnan(y_true_label).any() or np.isnan(y_score_label).any():
            print(f"Warning: NaN detected in label {label}")
            if np.isnan(y_true_label).any():
                print(f"y_true for label {label}: {y_true_label}")
                raise RuntimeError("Nan in y_true_label")
            if np.isnan(y_score_label).any():
                print(f"y_score for label {label}: {y_score_label}") 
                raise RuntimeError("Nan in y_score_label")
          else :
            auprc = average_precision_score(y_true_label, y_score_label)
            individual_auprc.append(auprc)
      overall_auprc = np.mean(individual_auprc)
      return individual_auprc,overall_auprc
    
    def transform_data_for_eval_each_disease(self, predictions_tensor = None, target_tensor = None):
      predict_label_dim = predictions_tensor.shape[-1]
      batch_n = predictions_tensor.shape[0]
      disease_n = predictions_tensor.shape[1]
      transform_data = {}
      if predict_label_dim == 1: # binary classification
        predictions_tensor = predictions_tensor.view(batch_n, -1)
        for disease in range(disease_n):
          each_disease_transform_data = {}
          disease_pred = predictions_tensor[:, disease].cpu().numpy().astype(np.float64)
          disease_target = target_tensor[:, disease].cpu().numpy().astype(np.int32)
          each_disease_transform_data["pre"] = disease_pred
          each_disease_transform_data['tar'] = disease_target
          transform_data[constants.CHEXPERT_LABELS[disease]] = each_disease_transform_data
      return transform_data
      
    def evaluate(self, eval_dataloader=None, training_step = 0, dump = {}):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_tensor =  torch.empty(0).to(self.device)
        label_tensor =  torch.empty(0).to(self.device)
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            data = {key: value.to(self.device) for key, value in data.items()}
            with torch.no_grad():
                _, classifier_out = self.clf(**data, eval=True)
                pred = classifier_out['logits']  
            pred_tensor = torch.cat((pred_tensor, pred), dim=0)  
            label_tensor = torch.cat((label_tensor, (data["img_labels"])), 0)     
        outputs = {'pred':pred_tensor, 'labels':label_tensor, "loss": classifier_out['loss_value']}
        num_batch = pred_tensor.shape[0]
        dump = {"dump_path": "/home_data/home/v-liudsh/coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/dump/"}
        if dump != None and "dump_path" in dump:
          tensor_dict = {"predictions": pred_tensor.reshape(num_batch,-1), "labels": label_tensor}
          pwd = os.getcwd()
          dump_path = dump["dump_path"]
          if not os.path.exists( pwd + "/" + dump_path): os.makedirs( pwd + "/" + dump_path)
          print(f"save tensor_dict in {pwd + dump_path}")
          torch.save(tensor_dict, pwd + "/" + dump_path + "/tensor.pth")
        
        # Evaluation metric depends on the labelling strategy used
        if self.mode == 'binary':
            if pred.shape[-1] == 1:    # NN output 1 value for binary classification
              pred_tensor =  torch.sigmoid(pred_tensor)
              transform_data = self.transform_data_for_eval_each_disease(pred_tensor, label_tensor)
              each_auc = {}
              for i, disease in enumerate(constants.CHEXPERT_LABELS):
                auc = roc_auc_score(transform_data[disease]["tar"] , transform_data[disease]["pre"],)
                each_auc[disease] = auc
              outputs["auc_dict"] = each_auc["disease_auc"]

            else: # have 2 outputs
                if pred.shape[-1] == len(constants.CHEXPERT_LABELS):
                  print("multi-labels classification.")
                  pred_tensor =  torch.sigmoid(pred_tensor)

                  # metric = calculate_metrics(pred = pred_tensor, target = label_tensor)
                  auc_metric = calculate_auc(pred = pred_tensor, target = label_tensor)
                  outputs["auc_dict"] = auc_metric

                  if torch.isnan(pred_tensor).any():
                    print("The pred_tensor contains NaN values")
                    print(f"pred_tensor: {pred_tensor}")
                    print(f"sigmoid(pred_tensor): {torch.sigmoid(pred_tensor)}")
                    print (f"the shape of torch.sigmoid(pred_tensor): {torch.sigmoid(pred_tensor).shape}")
                  metric = self.calculate_metrics(predict_proba = pred_tensor, target = label_tensor)
                  individual_auprc,overall_auprc = self.get_AUPRC(pred_probs = pred_tensor, labels = label_tensor)
                  each_aucprc = {}
        

                  # 通过循环将 key 和 value 组合为字典
                  for key, value in zip(self.labels, individual_auprc):
                      each_aucprc[f"{key}_auprc"] = value
                  outputs["overall_auprc"] = overall_auprc
                  outputs.update(each_aucprc)
                  outputs.update(metric)

                # cnf_matrix = confusion_matrix(labels, pred_label)
                # res = self.process_confusion_matrix(cnf_matrix)
                # outputs.update(res)

            # res = classification_report(labels, pred_tensor., output_dict=True, zero_division=np.nan_to_num)
            # res = res['macro avg']
            # res.pop('support')
            # outputs.update(res)

        elif self.mode == 'multiclass':
            auc_dict = self.get_AUC(pred_tensor.reshape(num_batch,-1), label_tensor)

            pred_tensor = pred_tensor.reshape(num_batch, len(constants.CHEXPERT_LABELS), 3)
            pred_tensor = pred_tensor.argmax(-1)
            acc = (pred_tensor == label_tensor).sum()/pred_tensor.numel()
            outputs['acc'] = acc
            outputs['auc_dict'] = auc_dict
            
            ### in my case, using AUC as the major metric
            ### during training phase, this operation work in CPU which slow the calculation.
            # res = classification_report(label_tensor.flatten().cpu(), pred_tensor.flatten().cpu(), output_dict=True)
            # res = res['macro avg']
            # res.pop('support')
            # outputs.update(res)

            # cnf_matrix = confusion_matrix(label_tensor.flatten().cpu(), pred_tensor.flatten().cpu())
            # res = self.process_confusion_matrix(cnf_matrix)
            # outputs.update(res)
            ###
        
        return outputs
    
    def evaluate_testing(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        overall_logits_list = []
        overall_label_list = []
        overall_prediction_list = []
        accumu_acc = []
        results = []
        pred_list = []
        label_list = []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            # pred_list = []
            # label_list = []
            with torch.no_grad():
                _, classifier_out, _ = self.clf(**data)
                pred = classifier_out['logits']
            pred_list.append(pred)
            
            tmp = [json.loads(i) for i in data["img_labels"]]
            label_list.append(torch.tensor(tmp))
            pred_list_t = torch.cat(pred_list, 0)
            labels = torch.cat(label_list, 0).cpu().detach().numpy()
            overall_label_list.append(labels)

            pred = pred_list_t.cpu().detach().numpy()     
            outputs = {'pred':pred, 'labels':labels}

            if self.mode is None:
                if len(labels.shape) == 1:
                    if len(np.unique(labels)) == 2:
                        self.mode = 'binary'
                    else:
                        self.mode = 'multiclass'
                else:
                    self.mode = 'multilabel'
                print(f'no mode specified, will pick mode `{self.mode}` by data.')

            if self.mode == 'binary':
                if pred.shape[1] == 1:
                    pred_score = torch.tensor(pred).sigmoid().numpy().flatten()
                    auc = roc_auc_score(labels, pred_score)
                    outputs['auc'] = auc
                    pred_label = np.ones(len(pred))
                    pred_label[pred_score<0.5] = 0
                    acc = (pred_label == labels).mean()
                    outputs['acc'] = acc
                    

                else: # have 2 outputs
                    pred_score = torch.tensor(pred).softmax().numpy()
                    pred_label = np.argmax(pred_score, 1)
                    acc = (pred_label == labels).mean()
                    outputs['acc'] = acc

                    # cnf_matrix = confusion_matrix(labels, pred_label)
                    # res = self.process_confusion_matrix(cnf_matrix)
                    # outputs.update(res)

                res = classification_report(labels, pred_label, output_dict=True, zero_division=np.nan)
                res = res['macro avg']
                res.pop('support')
                outputs.update(res)

            if self.mode == 'multiclass':
                # if len(pred.shape) == 2:
                #     pred = pred.view()
                # print(labels)
                # print(f"the shape of labels: {labels.shape}")
                num_batch = pred.shape[0]
                pred = pred.reshape(num_batch, len(constants.CHEXPERT_LABELS), 3)
                overall_logits_list.append(pred)
                pred_label = pred.argmax(-1)
                overall_prediction_list.append(pred_label)
                outputs["pred_label"] = pred_label
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc
                accumu_acc.append(acc)
                res = classification_report(labels.flatten(), pred_label.flatten(), output_dict=True, zero_division=np.nan)
                # print(res)
                res = res['macro avg']
                res.pop('support')
                outputs.update(res)
            
            if self.mode == 'multilabel':    ## focus on multi-labels
                pred_score = torch.tensor(pred).sigmoid().numpy()
                auroc_list, auprc_list,mse = [], [],[]
                for i in range(pred_score.shape[1]):
                    y_cls = labels[:, i]
                    pred_cls = pred_score[:, i]
                    # print("Y_CLS: ",y_cls, "\n")
                    # print("pred_cls: ",pred_cls, "\n")
                    # auprc_list.append(average_precision_score(y_cls, pred_cls, pos_label=1))
                    mse.append(mean_squared_error(y_cls, pred_cls))
                    # auroc_list.append(roc_auc_score(y_cls, pred_cls))
                outputs['auc/mse'] = np.mean(mse)
                outputs['auprc'] = -99 #np.mean(auprc_list)
            results.append(outputs)

        # overall_label_list = np.hstack(overall_label_list)
        overall_label_list = np.concatenate(overall_label_list, axis=0)
        overall_logits_list = np.concatenate(overall_logits_list, axis=0)
        overall_prediction_list = np.concatenate(overall_prediction_list, axis=0)
        # overall_logits_list = np.hstack(overall_logits_list)
        # overall_prediction_list = np.hstack(overall_prediction_list)
        return {"result": results, "overall_logit":overall_logits_list, "overall_label":overall_label_list, "overall_prediction": overall_prediction_list, "accumul_acc": accumu_acc}
    

    def cnf_matrix(self, labels, pred_label):
        cnf_matrix = confusion_matrix(labels, pred_label)
        res = self.process_confusion_matrix(cnf_matrix)
        return res


    def process_confusion_matrix(self, cnf_matrix):
        print(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        outputs = {}
        # Sensitivity, hit rate, recall, or true positive rate
        outputs['tpr'] = TP/(TP+FN)
        # Specificity or true negative rate
        outputs['tnr'] = TN/(TN+FP) 
        # Precision or positive predictive value
        outputs['ppv'] = TP/(TP+FP)
        # Negative predictive value
        outputs['npv'] = TN/(TN+FN)
        # Fall out or false positive rate
        outputs['fpr'] = FP/(FP+TN)
        # False negative rate
        outputs['fnr'] = FN/(TP+FN)
        # False discovery rate
        outputs['fdr'] = FP/(TP+FP)

        # Overall accuracy for each class
        outputs['acc1'] = (TP+TN)/(TP+FP+FN+TN)
        if cnf_matrix.shape[0] > 2: # multiclass
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = np.mean(v)
        else:
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = v[1]
        return outputs

    def get_AUC(self, predictions_tensor, labels_tensor, plot=False, record_roc = False, training_step = 0, labeling_strategy = None):
      """
      for each label(disease) gets its own auc
      plot: bool, whether plot the roc plot in this function
      return auc value 
      """
      disease_auc = {}
      bins = [i/20 for i in range(20)] + [1]
      for i, disease in enumerate(constants.CHEXPERT_LABELS):
        label_dis = labels_tensor[:, i]
        each_class_roc = {}
        for k, j in enumerate(constants.class_name):
          pred_dis = predictions_tensor[:, i*len(constants.class_name) + k].cpu().numpy()
          true_class = [1 if constants.class_name[j] == y else 0 for y in label_dis]
          if(len(set(true_class))==1):
            print(constants.RED, "this disease have something wrong: "+constants.RESET, disease, ", ", j, "in this case set auc is 0!!!")
            each_class_roc[j] = 0
            continue
          # self.plot_(true_class, pred_dis)
          each_class_roc[j] = roc_auc_score(true_class, pred_dis, multi_class="ovr", average="micro",)
        disease_auc[disease] = each_class_roc
      return disease_auc
        
        
    def get_all_roc_coordinates(self, y_real, y_proba):
      '''
      Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
      
      Args:
          y_real: The list or series with the real classes.
          y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
          
      Returns:
          tpr_list: The list of TPRs representing each threshold.
          fpr_list: The list of FPRs representing each threshold.
      '''
      tpr_list = [0]
      fpr_list = [0]
      for i in range(len(y_proba)):
          threshold = y_proba[i]
          y_pred = y_proba >= threshold
          tpr, fpr = self.calculate_tpr_fpr(y_real, y_pred)
          tpr_list.append(tpr)
          fpr_list.append(fpr)
      return tpr_list, fpr_list
        
    def plot_roc_curve(self, tpr, fpr, scatter = True, ax = None):
      '''
      Plots the ROC Curve by using the list of coordinates (tpr and fpr).
      
      Args:
          tpr: The list of TPRs representing each coordinate.
          fpr: The list of FPRs representing each coordinate.
          scatter: When True, the points used on the calculation will be plotted with the line (default = True).
      ''' 
      if ax == None:
          plt.figure(figsize = (5, 5))
          ax = plt.axes()
      
      if scatter:
          sns.scatterplot(x = fpr, y = tpr, ax = ax)
      sns.lineplot(x = fpr, y = tpr, ax = ax)
      sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
      plt.xlim(-0.05, 1.05)
      plt.ylim(-0.05, 1.05)
      plt.xlabel("False Positive Rate")
      plt.ylabel("True Positive Rate")
      plt.savefig('./output/AUC_img/output_plot.png')
          
    def plot_(self, true_label, prob):
        plt.figure(figsize = (14, 3)) 
        # ax = plt.subplot(2, 3, i+1)
        # sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        # ax.set_title(c)
        # ax.legend([f"Class: {c}", "Rest"])
        # ax.set_xlabel(f"P(x = {c})")
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        # ax_bottom = plt.subplot(2, 3, i+4)
        tpr, fpr = self.get_all_roc_coordinates(true_label, prob)
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        # ax_bottom.set_title("ROC Curve OvR")
        
        # Calculates the ROC AUC OvR
        print(disease, j)
        print(f"pred: \n{pred_dis}")
        print(f"class: \n{true_class}, {len(true_class)}")
        
      
      
      
