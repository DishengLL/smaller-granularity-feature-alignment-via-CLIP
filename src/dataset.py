from collections import defaultdict
from typing import Union, List, Optional
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
import os
import json
import constants
from torch import tensor
import torch
import ast

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pwd = os.getcwd()
training_data_path = pwd + "/../data/project_using_data/all_train_data_3_11.csv"
testing_data_path = pwd + r"/../data/project_using_data/all_test_3_11.csv"
training_AP_PA_data_path = pwd + "/../data/project_using_data/all_train_data_4_12_AP_PA.csv"
testing_AP_PA_data_path = pwd + r"/../data/project_using_data/all_test_data_4_12_AP_PA.csv"


# dataset.py provide all the tensor model needs
class ImageTextContrastiveDataset(Dataset):
    '''
    using nntype to determine which backbone process text
    custom vision encoder using clip_preprocess images
    '''
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, source_data='p10_12_train.csv', imgtransform=None, prompt_type="basic", 
                 backbone_type = None, binary = False, **kwargs) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        filename :  the csv file contains all of training data
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        self.prompt_type = prompt_type
        if source_data is None:
            raise ValueError("source_data should be specified, which indicates the path of original data")
        if kwargs.get("AP_PA_view"):
          filename = training_AP_PA_data_path
        else:  
          filename = training_data_path
        # filename = pwd+"/../data/project_using_data/train_3_15.csv"  # temp small training data 10000
        print(constants.RED + 'load training data from' + constants.RESET, filename)
        self.df = pd.read_csv(filename, index_col=0)
        self.df = self.df.head(100000)
        if backbone_type not in ["clip", "biomedclip", "custom", "biovil-t", "cxr-bert-s"]:
            raise ValueError("backbone type error")
        if backbone_type == "clip" and prompt_type == "basic":
            print( constants.RED + f"{backbone_type} text encoder" + constants.RESET + " processes " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/clip_basic.pt"
        elif backbone_type == "biomedclip":
          if prompt_type == "basic":
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/biomedclip_basic.pt"
          elif prompt_type == "dis_diag_des":
            self.prompts_tensor_path = pwd + r"/../constants/text_embeddings/biomed_dia_des_text_embeddings.pt"
          else:
            raise NotImplementedError()
            print( constants.RED + f"{backbone_type} text encoder" + constants.RESET + " processes " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
        elif backbone_type == "cxr-bert-s" and prompt_type == "basic":
            print( constants.RED + f"{backbone_type} text encoder" + constants.RESET + " processes " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/cxr_bert_s.pt"          
        elif backbone_type == "biovil-t" and prompt_type == "basic":
            print( constants.RED + f"{backbone_type} text encoder" + constants.RESET + " processes " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/biovil_t.pt"
        else:
            print(f"Custom your prompts!! Attention!!!!!! {prompt_type}, {backbone_type}")
            raise ValueError()
        self.backbone = backbone_type
        self.binary = binary
        self.labeling_strategy = None
        if "labeling_strategy" in kwargs:
          self.labeling_strategy = kwargs["labeling_strategy"]
        self.prompt_tensor = torch.load(self.prompts_tensor_path)
          
    def find_corresponding_text_embedding(self, disease:int = None, diagnose = None):
      # disease_list = constants.CHEXPERT_LABELS
      # disease_index = disease_list.index(disease)
      diagnose_index = 0 if diagnose == 1 else 1
      return disease * 2 + diagnose_index
      
    def convert_labels_2_tensor(self, string_representation:str):
      # numbers_list = [int(num) for num in string_representation[1:-1].split(', ')]
      # tensor_representation = tensor(numbers_list)
      list_value = eval(string_representation)
      return tensor(list_value)
      # list_value_str = string_representation[string_representation.index('['):string_representation.rindex(']')+1]
      # list_value = ast.literal_eval(list_value_str)
      # tensor_representation = tensor(list_value)
      # return tensor_representation

    def __getitem__(self, index):
      '''
      return 
      image path: contain image tensor
      prompt_path: contain prompt tensor
      labels: tensor
      '''
      row = self.df.iloc[index]
      if self.backbone == "biomedclip":
        img_tensor_path =  row.BiomedClip_img_tensor_path
      elif self.backbone == "clip":
        img_tensor_path =  row.Clip_img_tensor_path
      elif self.backbone == "biovil-t" or self.backbone == "cxr-bert-s":
        img_tensor_path = row.Biovil_img_tensor_path_old
      else: 
        raise NotImplemented(f"backbone model type error {self.backbone}")
      img_tensor = torch.load(img_tensor_path)

    
      if self.labeling_strategy == "S1":
        label = self.convert_labels_2_tensor(row.strategy1_14_labels)
        if self.prompt_type == "dis_diag_des":
          """
          this prompt type : template(disease, diagnoses, description)
          """
          indices = []
          for disease, diagnose in enumerate(label):
            indices.append(self.find_corresponding_text_embedding(disease, diagnose))
          return img_tensor, self.prompt_tensor[indices], label
        return img_tensor, self.prompt_tensor, label
      return img_tensor, self.prompt_tensor, self.convert_labels_2_tensor(row.project_3_classes_14_labels)

    def __len__(self):
        return len(self.df)

class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs["img"].append(data[0])
            inputs["prompts"].append(data[1])
            inputs["img_labels"].append(data[2]) 

        inputs['img'] = torch.tensor(np.stack(inputs["img"])).to(device)
        inputs['prompts'] = torch.stack(inputs["prompts"]).to(device)
        inputs['img_labels'] = torch.stack(inputs["img_labels"]).to(device)
        return inputs

class ImageTextContrastiveDataset1(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, source_data='p10_12_train.csv', imgtransform=None, prompt_type=None) -> None:
        '''support data list in mimic-cxr-train, chexpert-train
        filename :  the csv file contains all of training data
        '''
        super().__init__()
        if source_data is None:
            raise ValueError("source_data should be specified, which indicates the path of original data")
        filename = pwd + "/../data/project_using_data/all_train_data_sub_3_14.csv"

        print(constants.RED + 'load training data from' + constants.RESET, filename)
        self.df = pd.read_csv(filename, index_col=0)
        if prompt_type is None:
            self.prompts = constants.BASIC_PROMPT
        else:
            raise ValueError("Custom your prompts!! Attention!!!!!!")

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path =  row.ws_file_path
        return img_path, self.prompts, row.project_3_classes_14_labels

    def __len__(self):
        return len(self.df)

class ImageTextContrastiveCollator1:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        for data in batch:
            inputs['img'].append(data[0])
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
        inputs['prompts'] =  report_list
        return inputs

class TestingDataset(Dataset):
    def __init__(self,
        datalist=['testing'],  # specify the df which used in testing 
        prompt_type="basic",
        backbone_type = None,
        **kwargs
        ) -> None:
        '''
        using data in the datalist to be testing data
        '''
        super().__init__()
        # if prompt_type is None:
        #     self.prompts = constants.BASIC_PROMPT
        # else:
        #     raise NotImplementedError("Custom your prompts!! Attention!!!!!! ToDo: define new prompt in constants.py file")
        self.backbone = backbone_type
        if kwargs.get("AP_PA_view"):
          filename = testing_AP_PA_data_path
        else:
          filename = testing_data_path
        print(constants.RED + 'Testing load testing data from' + constants.RESET, filename)
        self.df = pd.read_csv(filename, index_col=0)
        # self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        if backbone_type not in ["clip", "biomedclip", "custom", "densenet", "cxr-bert-s", "biovil-t"]:
            raise ValueError("backbone type error: {backbone_type}")
        if backbone_type == "biomedclip" and prompt_type == "basic":
            print( "currently using " + constants.RED + f"{backbone_type}" + constants.RESET + " to process " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/biomedclip_basic.pt"
        elif backbone_type == "clip" and prompt_type == "basic":  # CLIP and custom (densenet) using CLIP images preprocess
            print( "currently using " + constants.RED + f"{backbone_type}" + constants.RESET + " to process " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/clip_basic.pt"
        elif backbone_type == "cxr-bert-s" and prompt_type == "basic":
            print( "currently using " + constants.RED + f"{backbone_type}" + constants.RESET + " to process " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
            self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/cxr_bert_s.pt"          
        elif backbone_type == "biovil-t" and prompt_type == "basic":
           print( "currently using " + constants.RED + f"{backbone_type}" + constants.RESET + " to process " + constants.RED + f"{prompt_type}" + constants.RESET + " prompt")
           self.prompts_tensor_path = pwd + r"/../data/prompts_tensors/basic/biovil_t.pt"
        else:
            print(f"Custom your prompts!! Attention!!!!!! {prompt_type}, {backbone_type}")
            raise ValueError()
        self.labeling_strategy = None
        if "labeling_strategy" in kwargs:
          self.labeling_strategy = kwargs["labeling_strategy"]
    
    def convert_labels_2_tensor(self, string_representation:str):
      # numbers_list = [int(num) for num in string_representation[1:-1].split(', ')]
      # tensor_representation = tensor(numbers_list)
      # return tensor_representation
      numbers_list = eval(string_representation)
      return tensor(numbers_list)


    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.backbone == "biomedclip":
          img_tensor_path =  row.BiomedClip_img_tensor_path
        elif self.backbone == "clip":
          img_tensor_path =  row.Clip_img_tensor_path
        elif self.backbone == "biovil-t" or self.backbone == "cxr-bert-s":
          img_tensor_path = row.Biovil_img_tensor_path
        else:  
          raise NotImplemented(f"backbone model type error {self.backbone}")
          ## default: using clip image preprocessing
        img_tensor = torch.load(img_tensor_path)
        prompt_tensor = torch.load(self.prompts_tensor_path)
        if self.labeling_strategy == "S1":
          return img_tensor, prompt_tensor, self.convert_labels_2_tensor(row.strategy1_14_labels)
        return img_tensor, prompt_tensor, self.convert_labels_2_tensor(row.project_3_classes_14_labels)

    def __len__(self):
        return len(self.df)

class TestingCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs["img"].append(data[0])
            inputs["prompts"].append(data[1])
            inputs["img_labels"].append(data[2]) 
        inputs['img'] = torch.tensor(np.stack(inputs["img"])).to(device)
        inputs['prompts'] = torch.stack(inputs["prompts"]).to(device)
        inputs['img_labels'] = torch.stack(inputs["img_labels"]).to(device)
        return inputs

### the classes below focus on data exploration and raw usable dataset maker
class data_exploration():
    def __init__(self, 
                 file_path:str = None):
        if file_path == None:
            raise ValueError("the source file path should be specified!")
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path, index_col=False)

    def info(self,):
        shape = self.df.shape
        columns = self.df.columns
        rows = self.df.index
        print(f"the shape of {self.file_path} is:\n{shape}")
        print(f"the columns of df: {columns}")
        print(f"the rows of df: {rows}")
        return (shape, columns, rows)

    def browse(self, n = 10):
        print(self.df.head(n))

    def browse_condition(self, target:str = None, column_name:str = None):
        if target is None or column_name is None:
            raise ValueError("the target and column name should be set, which indicates you search condition!")
        condition = self.df[column_name] == target
        print(self.df[condition])

    def get_df (self):
        return self.df

class Process_raw_csv():
    def __init__(self):
        print("run this class to preprocess raw csv file")
        

    def generate_label(self, row):
        label_dic = {}
        study_id = -1
        for disease in constants.CHEXPERT_LABELS:
            label_dic[disease] = constants.UNCERTAIN_CLASS
        for column_name, value in row.items():
            if column_name == "study_id": 
              study_id = int(value)
              continue
            # if column_name == "No Finding" and value == constants.POSITIVE:
            #     label_dic[column_name] = constants.POSITIVE_CLASS
            #     label_dic['study_id'] = study_id
            #     # return label_dic
            if value == constants.POSITIVE:
                label_dic[column_name] = constants.POSITIVE_CLASS
                if column_name != "No Finding":
                  label_dic["No Finding"] = constants.NEGATIVE_CLASS
            elif value == constants.NEGATIVE:
                label_dic[column_name] = constants.NEGATIVE_CLASS
                if column_name != "No Finding":
                  label_dic["No Finding"] = constants.NEGATIVE_CLASS
            
        label_dic['study_id'] = study_id
        return label_dic

    def image_label_preprocess(self, df):
    ### 按照病人进行排列, 返回的df包含两列： study_id, label
        df = df[["study_id"] + constants.CHEXPERT_LABELS]
        df['label'] = df.apply(self.generate_label, axis=1)
        return df[["study_id", "label"]]

    def mapping(self, _map_):
        labels = []
        if isinstance(_map_, str):
            _map_ = _map_.replace("'", '"')
        if not isinstance(_map_, dict):
            _map_ = json.loads(_map_)
        for _, value in _map_.items():
            labels.append(value)
        return labels

    def path_concat(self, _df_):
        img_path = "/"+"p"
        dir_path = _df_["subject_id"].map(str)
        _df_['dir'] = dir_path.apply(lambda x: x[:2])
        sub_path = "/p" + _df_['dir'] + "/" + img_path + _df_["subject_id"].map(str)+'/' + "s" + _df_["study_id"].map(str) + "/"
        img_path = constants.DATA_DIR + sub_path
        _df_["img_path"] = img_path

    def raw_data_process(self, save_path=pwd+"\\data\\mimic-cxr-train\\cxr_postprocess_2_Nov.csv"):
        '''
        the main method in this class
        process raw data sheet.
        the generated one has col [subject_id, study_id, label, img_path, train_label]
        the 
        '''
        df = pd.read_csv(pwd+'\data\\mimic-cxr-2.0.0-chexpert.csv', index_col = 0)
        df = self.image_label_preprocess(df)
        df.reset_index(inplace=True)
        self.path_concat(df)
        df["train_label"] = df["label"].apply(self.mapping)
        df.to_csv(save_path) 

class get_sub_set_data():
    def __init__(self, n = 7, source=None, condition = None, col_name = None):
        self.num = n
        if source is None:
            raise ValueError("Source is the original dataset, which should be specified!!")
        self.source = source
        self.condition = condition
        self.col_name = col_name

    def extract_by_condition(self, dest=None):
        if self.col_name is None:
            raise ValueError("you should specify one column")
        df = pd.read_csv(self.source)
        if self.condition is None:
            raise ValueError("condition error!")
        filtered_data = df[df[self.col_name].apply(self.condition)]

        # 将筛选后的数据保存到新的CSV文件
        filtered_data.to_csv(dest, index=False)
    
    def get_sub_set(self, save = False, cols:list = None, row = None, dist = None):
        if self.condition is not None:
            self.extract_by_condition(dist)
            return
        
        if row is None:
            row = self.num
        source = pd.read_csv(self.source)
        if cols is None:
            target = source.iloc[:row, :]
        else:
            target = source.iloc[:row, cols]
        if save is True:
            target.to_csv(dist, index=False)

        return target   

def condition(row):   
    """
    just using data of p10-p12
    """
    return row in [10, 11, 12]

class make_data_set():
    '''
    construct this final dataset which can be used in concrete training and testing
    --- final.csv:  contains both testing and training data
    --- train_only.csv: subset of final.csv, contains all of training data
    --- test_only.csv: subset of final.csv, contains all of testing data
    '''
    def __init__(self, source1 = "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\data\\MIMIC_CXR\\physionet.org\\files\\mimic-cxr-jpg\\2.0.0\\mimic-cxr-2.0.0-split.csv", 
                 source2 = "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\p10_12.csv"):
        if source1 is None or source2 is None:
            raise ValueError("source1 and source2 should be defined")
        self.df1 = pd.read_csv(source1)
        self.df2 = pd.read_csv(source2)
        self.split_dataset = self.df1
        self.basic_dataset = self.df2
        self.count = 0
        self.interval = 1000
    
    def check_ext(self, string: str, split = ".", target = "jpg"):
        return string.split(split)[-1] == target

    def get_images(self, df) -> list:
        img_dir = df["img_path"]
        try:
            file_and_dir = os.listdir(img_dir)
        except:
            files = ["Void"]
            print(img_dir)
            return files
            raise ValueError("wrong path!!!!!")
        files = [img_dir + "/" + image for image in file_and_dir if self.check_ext(img_dir + "/" + image)]
        return files
    
    def does_train(self, L_str: List[str] = None) -> bool:
        does_train = []
        if (L_str[0] == "Void"):
            return ["Void"]
        for i in L_str:
            tmp = i.split("/")
            subj = tmp[-4][1:]
            stud = tmp[-3][1:]
            id = tmp[-1].split(".")[0]
            condition_1 = self.split_dataset["dicom_id"] == id
            condition_2 = self.split_dataset["study_id"] == int(stud)
            condition_3 = self.split_dataset["subject_id"] == int(subj)
            result = self.split_dataset[ (condition_1 & condition_2) & condition_3]
            result = result.split

            does_train.append(result)
        return does_train
    
    def combine(self, save = False, path=None):
        new_df = pd.DataFrame(columns=["subject_id","study_id", "label", "img_path", "train_label", "file_path", "split"])
        if save == True and path is not None:
            new_df.to_csv(path, mode='w', header=True, index=False)
        for _, row in self.df2.iterrows():
            files = self.get_images(row)
            trains_p = self.does_train(files)
            if files[0] == "Void":
                continue

            else:
                for i, j in enumerate(files):
                    self.count += 1
                    new_df = pd.concat([new_df, pd.DataFrame({'subject_id': row['subject_id'], 'study_id': row['study_id'], "label": row["label"], 
                                        "img_path": row["img_path"], "train_label": row["train_label"], 'file_path': j,  'split': trains_p[i]})], ignore_index=True)
                    if (self.count % self.interval == 0) and save == True and path is not None:
                        new_df.to_csv(path, mode='a', header=False, index=False)
                        new_df = pd.DataFrame(columns=["subject_id","study_id", "label", "img_path", "train_label", "file_path", "split"])
                        print(f"the current image number: {self.count}")
        return new_df
    
    def save_training_testing(self, source = None, target_1 = None, target_2 = None, target_val = None):
        if source is None or target_1 is None or target_2 is None or target_val is None:
            raise ValueError("input parameter(s) is(are) problematic!")
        df_source = pd.read_csv(source)
        condition = df_source["split"] == "train"
        df = df_source[condition]
        df.to_csv(target_1)
        condition = df_source["split"] == "test"
        df = df_source[condition]
        df.to_csv(target_2)
        condition = df_source["split"] == "validate"
        df = df_source[condition]
        df.to_csv(target_val)
        del(df)
        
        
                
        # new_df.to_csv("/Users/liu/Desktop/school_academy/ShanghaiTech/learning/code/diagnosisP/x_ray_constrastive/data/final_data.csv")

# if __name__ == "__main__":
    
#     a = Process_raw_csv()
#     a.raw_data_process()
#     # b = get_sub_set_data(source="D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\cxr_postprocess_2_Nov.csv", condition=condition, col_name="dir")
#     # b.get_sub_set(save=True, dist="D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\p10_12.csv")

#     print("----making train/test dataset-----\n")
#     # source2="/Users/liu/Desktop/school_academy/ShanghaiTech/learning/code/diagnosisP/x_ray_constrastive/data/mimic-cxr-train/cxr_postprocess.csv"
#     c = make_data_set()
#     x = c.combine(save=True, path="D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\P10_12_final.csv")
#     # print(x)
#     print(x[x["split"] == "test"].sum())
#     print("--------get training and testing dataset csv file---------\n")
#     c = make_data_set()
#     c.save_training_testing(source="D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\P10_12_final.csv", target_1 = "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\P10_12_train.csv", target_2 = "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\P10_12_test.csv", target_val = "D:\\exchange\\ShanghaiTech\\learning\\code\\diagnosisP\\x_ray_constrastive\\data\\mimic-cxr-train\\P10_12_validate.csv")


