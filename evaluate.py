import pdb

import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from tqdm import tqdm
import json

import constants

class Evaluator:
    '''do evaluation on chexpert5x200 zero-shot classification
    '''
    def __init__(self,
        FG_model_cls,
        eval_dataloader=None,
        mode=None,
        ) -> None:
        '''specify class_names if doing zero-shot classification.
        mode: `binary`, 'multiclass`, or `multilabel`,
        if set None, the method will automatically decide from data.
        recommend to set explicitly to avoid errors.
        '''
        self.clf = FG_model_cls
        self.mode = mode
        self.eval_dataloader = eval_dataloader
    
    def evaluate(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                
        #                 a = self.Contrastive_Model(input_text, img_path)
        # b = self.PN_Classifier(a['img_embeds'], img_labels)
        # c = self.Orthogonal_dif(a['text_embeds'])
        # return a, b, c
                branch_out, classifier_out, _ = self.clf(**data)
                pred = classifier_out['logits']
                # print(f"the shape of pred: {pred.shape}")
            pred_list.append(pred)
            # print("\n t: ",data['img_labels'],"\n")
            tmp = [json.loads(i) for i in data["img_labels"]]
            label_list.append(torch.tensor(tmp))
            # nested_list = [json.loads(s) for s in img_label]
            # img_label = torch.tensor(np.stack(nested_list), dtype=torch.float32)
        # print("label_list:", label_list)
        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = pred_list.cpu().detach().numpy()        
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
                pred_score = torch.tensor(pred).sigmoid().numpy()
                pred_label = np.argmax(pred_score, 1)
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc

                # cnf_matrix = confusion_matrix(labels, pred_label)
                # res = self.process_confusion_matrix(cnf_matrix)
                # outputs.update(res)

            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

        if self.mode == 'multiclass':
            # if len(pred.shape) == 2:
            #     pred = pred.view()
            print(labels)
            print(f"the shape of labels: {labels.shape}")
            num_batch = pred.shape[0]
            pred = pred.reshape(num_batch, 13, 3)

            pred_label = pred.argmax(-1)
            print(pred_label)
            acc = (pred_label == labels).mean()
            outputs['acc'] = acc
            res = classification_report(labels.flatten(), pred_label.flatten(), output_dict=True)
            print(res)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

            # cnf_matrix = confusion_matrix(labels, pred_label)
            # res = self.process_confusion_matrix(cnf_matrix)
            # outputs.update(res)
        
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
        return outputs
    
    def process_confusion_matrix(self, cnf_matrix):
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
        # outputs['acc'] = (TP+TN)/(TP+FP+FN+TN)
        if cnf_matrix.shape[0] > 2: # multiclass
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = np.mean(v)
        else:
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = v[1]
        return outputs
