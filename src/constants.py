'''drawn from Gloria github: https://github.com/marshuang80/gloria
'''

BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

POSITIVE = 1  # setting of the original dataset
NEGATIVE = 0  # setting of the original dataset
UNCERTAIN = -1  # setting of the original dataset

POSITIVE_CLASS = 0
NEGATIVE_CLASS = 1
UNCERTAIN_CLASS = 2

class_name = ["POSITIVE", "NEGATIVE", "UNCERTAIN_CLASS"]


class_ratio_in_training = (
 ('Pleural Other', 0.008568868851511473),
 ('Fracture', 0.0196761266208119),
 ('Lung Lesion', 0.02721422653796505),
 ('Enlarged Cardiomediastinum', 0.03126023543918463),
 ('Pneumothorax', 0.04637496869159779),
 ('Consolidation', 0.0473623875305859),
 ('Pneumonia', 0.07026568791784675),
 ('Edema', 0.17726335664605128),
 ('Cardiomegaly', 0.19497427894341368),
 ('Lung Opacity', 0.22329634125195075),
 ('Pleural Effusion', 0.23415794848082),
 ('Atelectasis', 0.24406103693428124),
 ('Support Devices', 0.2995636090399399),
 ('No Finding', 0.33638710671830147)
 )


CHEXPERT_LABELS = [
 'Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Enlarged Cardiomediastinum',
 'Fracture',
 'Lung Lesion',
 'Lung Opacity',
 'No Finding',
 'Pleural Effusion',
 'Pleural Other',
 'Pneumonia',
 'Pneumothorax',
 "Support Devices"
]
NIH_LABELS = [  
              'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
              'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax' 
              ]

DIAGNOSTIC_CHEXPERT_LABELS = [
  'Atelectasis positive', 
  'Atelectasis negative', 
  'Cardiomegaly positive', 
  'Cardiomegaly negative', 
  'Consolidation positive', 
  'Consolidation negative', 
  'Edema positive', 
  'Edema negative', 
  'Enlarged Cardiomediastinum positive', 
  'Enlarged Cardiomediastinum negative', 
  'Fracture positive', 
  'Fracture negative', 
  'Lung Lesion positive', 
  'Lung Lesion negative', 
  'Lung Opacity positive',
  'Lung Opacity negative',
  'No Finding positive',
  'No Finding negative',
  'Pleural Effusion positive',
  'Pleural Effusion negative',
  'Pleural Other positive',
  'Pleural Other negative',
  'Pneumonia positive',
  'Pneumonia negative',
  'Pneumothorax positive',
  'Pneumothorax negative',
  'Support Devices positive', 
  'Support Devices negative'
  ]


BASIC_PROMPT = [
 'images for Atelectasis',
 'images for Cardiomegaly',
 'images for Consolidation',
 'images for Edema',
 'images for Enlarged Cardiomediastinum',
 'images for Fracture',
 'images for Lung Lesion',
 'images for Lung Opacity',
 'images for No Finding',
 'images for Pleural Effusion',
 'images for Pleural Other',
 'images for Pneumonia',
 'images for Pneumothorax',
 "images for Support Devices"
]

DESC_PROMPT = {
    "No Finding": "No pathological findings observed in the X-ray image; a normal result indicating the absence of detectable abnormalities or diseases.",
    "Enlarged Cardiomediastinum": "Enlargement of the heart and the structures in the central part of the chest, visible on a chest X-ray, often identified in the mediastinal region.",
    "Cardiomegaly": "Abnormal enlargement of the heart, potentially observed in the cardiac silhouette on a chest X-ray, indicating an increase in cardiac size.",
    "Lung Lesion": "An abnormality or injury in the lung tissue, which may include tumors, nodules, or other lesions, affecting specific areas within the lungs.",
    "Lung Opacity": "Increased density or lack of transparency in lung tissue observed on X-ray images, affecting various regions of the lungs and compromising visibility.",
    "Edema": "Accumulation of excess fluid in the body tissues, including the lungs, leading to swelling; in X-ray images, may manifest as increased density in the lung parenchyma.",
    "Consolidation": "Solidification of lung tissue, often due to conditions like pneumonia, affecting specific areas of the lungs and causing reduced air exchange.",
    "Pneumonia": "Inflammation of the lung tissue, typically affecting specific lobes or segments of the lungs, visible as infiltrates on X-ray images.",
    "Atelectasis": "Collapse or partial collapse of lung tissue, causing reduced air volume in specific lung regions, often observed as opacities on X-ray.",
    "Pneumothorax": "Presence of air in the pleural space, causing lung collapse; usually seen as a dark area along the pleural line on X-ray images.",
    "Pleural Effusion": "Accumulation of excess fluid in the pleural cavity, typically seen as blunting of the costophrenic angles or as fluid collections in specific pleural spaces on X-ray.",
    "Pleural Other": "Various abnormalities affecting the pleura, including pleurisy, pneumopleuritis, or pleural thickening; localized changes visible in the pleural region on X-ray.",
    "Fracture": "Break or crack in bones of the rib cage or sternum, often visible as discontinuities in specific ribs or bone structures on X-ray.",
    "Support Devices": "Presence of medical support devices in the patient, such as ventilators, tubes, or other supportive equipment; may be visible in specific areas of the chest or lung fields on X-ray."
}

Question_based_Prompt_p = [
 'How to diagnose atelectasis as positive from x-ray images?',
 'How to diagnose cardiomegaly as positive from x-ray images?',
 'How to diagnose consolidation as positive from x-ray images?',
 'How to diagnose edema as positive from x-ray images?',
 'How to diagnose enlarged cardiomediastinum as positive from x-ray images?',
 'How to diagnose fracture as positive from x-ray images?',
 'How to diagnose lung lesion as positive from x-ray images?',
 'How to diagnose lung opacity as positive from x-ray images?',
 'How to diagnose no finding as positive from x-ray images?',
 'How to diagnose pleural effusion as positive from x-ray images?',
 'How to diagnose pleural other as positive from x-ray images?',
 'How to diagnose pneumonia as positive from x-ray images?',
 'How to diagnose pneumothorax as positive from x-ray images?',
 'How to diagnose support devices as positive from x-ray images?']

Question_based_Prompt_n = [
 'How to diagnose atelectasis as negative from x-ray images?',
 'How to diagnose cardiomegaly as negative from x-ray images?',
 'How to diagnose consolidation as negative from x-ray images?',
 'How to diagnose edema as negative from x-ray images?',
 'How to diagnose enlarged cardiomediastinum as negative from x-ray images?',
 'How to diagnose fracture as negative from x-ray images?',
 'How to diagnose lung lesion as negative from x-ray images?',
 'How to diagnose lung opacity as negative from x-ray images?',
 'How to diagnose no finding as negative from x-ray images?',
 'How to diagnose pleural effusion as negative from x-ray images?',
 'How to diagnose pleural other as negative from x-ray images?',
 'How to diagnose pneumonia as negative from x-ray images?',
 'How to diagnose pneumothorax as negative from x-ray images?',
 'How to diagnose support devices as negative from x-ray images?']

CHEXPERT_DESCRIPTIONS = {
    'Atelectasis': {
        'positive': "Atelectasis is a condition in which the lung or a portion of it is collapsed or not fully inflated.",
        'negative': "No evidence of atelectasis is present in the lung."
    },
    'Cardiomegaly': {
        'positive': "Cardiomegaly refers to an enlarged heart. It can be caused by various conditions, including heart disease.",
        'negative': "The heart appears to be of normal size; no evidence of cardiomegaly is present."
    },
    'Consolidation': {
        'positive': "Consolidation is a condition in which the lung tissue becomes filled with fluid, such as pus or blood, making it appear more dense on imaging.",
        'negative': "No evidence of lung consolidation is present."
    },
    'Edema': {
        'positive': "Edema is the accumulation of fluid in the body's tissues, leading to swelling.",
        'negative': "No evidence of edema is present."
    },
    'Enlarged Cardiomediastinum': {
        'positive': "An enlarged cardiomediastinum refers to an abnormal enlargement of the heart and the structures in the middle of the chest (mediastinum).",
        'negative': "The size of the cardiomediastinal silhouette appears to be normal."
    },
    'Fracture': {
        'positive': "A fracture is a broken bone, which can be seen on imaging studies such as X-rays.",
        'negative': "No evidence of bone fracture is present."
    },
    'Lung Lesion': {
        'positive': "A lung lesion is an abnormal area in the lung tissue that may indicate the presence of a tumor or other abnormality.",
        'negative': "No evidence of lung lesion is present."
    },
    'Lung Opacity': {
        'positive': "Lung opacity refers to an area of increased density in the lung tissue, which may indicate the presence of fluid, inflammation, or other abnormalities.",
        'negative': "No evidence of lung opacity is present."
    },
    'No Finding': {
        'positive': "No finding indicates that no abnormality or disease is detected in the imaging study.",
        'negative': "Abnormalities or diseases are detected in the imaging study."
    },
    'Pleural Effusion': {
        'positive': "Pleural effusion is the accumulation of fluid in the pleural cavity, the space between the lungs and the chest wall.",
        'negative': "No evidence of pleural effusion is present."
    },
    'Pleural Other': {
        'positive': "Other pleural abnormalities or conditions.",
        'negative': "No evidence of other pleural abnormalities or conditions is present."
    },
    'Pneumonia': {
        'positive': "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.",
        'negative': "No evidence of pneumonia is present."
    },
    'Pneumothorax': {
        'positive': "Pneumothorax is a collapsed lung, which occurs when air leaks into the space between the lung and the chest wall.",
        'negative': "No evidence of pneumothorax is present."
    },
    'Support Devices': {
        'positive': "Support devices refer to medical equipment such as ventilators or feeding tubes used to support the patient's vital functions.",
        'negative': "No support devices are present."
    }
}


RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

## initial version
CHEXPERT_PROMPTS = CHEXPERT_LABELS

# DATA_DIR="D:/exchange/ShanghaiTech/learning/code/diagnosisP/x_ray_constrastive/data/mimic-cxr-train/"
DATA_DIR = "D:/project_x_ray_CLIP/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

COVID_TASKS = [
    'Normal',
    'COVID',
]
COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy','confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    }
}

RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]
RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    }
}

WEIGHTS_NAME = 'pytorch_model.bin'

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'

class_name = {"positive":0, "negative":1, "uncertain":2}


'''
referencing the experiments results derived from validation dataset in the work of [CheXclusion: Fairness gaps in deep chest X-ray classifiers](https://github.dev/LalehSeyyed/CheXclusion/blob/main/MIMIC/Actual_TPR.py), the thresholds from each labels should be adapted for the best performance.
'''
CheXpert_thresholds = {
  'Atelectasis': 0.1715822160243988,
  'Cardiomegaly': 0.25941015779972076,
  'Consolidation': 0.12079081088304515,
  'Edema': 0.29250548481941224,
  'Enlarged Cardiomediastinum': 0.11519031822681422,
  'Fracture': 0.1463069587945938,
  'Lung Lesion': 0.1433594509959221,
  'Lung Opacity': 0.3219067692756653,
  'No Finding': 0.2351817488670349,
  'Pleural Effusion': 0.32850325107574463,
  'Pleural Other': 0.13745994269847867,
  'Pneumonia': 0.08874680846929543,
  'Pneumothorax': 0.28035772740840914,
  'Support Devices': 0.43483529686927797}


MIMIC_CXR_thresholds = {
  'Airspace Opacity': 0.2067894160747527,
  'Atelectasis': 0.2571824133396149,
  'Cardiomegaly': 0.25030514299869533,
  'Consolidation': 0.13011438995599744,
  'Edema': 0.2605415195226669,
  'Enlarged Cardiomediastinum': 0.08953001797199245,
  'Fracture': 0.08186926990747445,
  'Lung Lesion': 0.13840098977088924,
  'No Finding': 0.36988158226013185,
  'Pleural Effusion': 0.37177377939224243,
  'Pleural Other': 0.07955108284950249,
  'Pneumonia': 0.14167169332504273,
  'Pneumothorax': 0.20276318490505213,
  'Support Devices': 0.38156365752220156
  }


NIH_thresholds = {
  'Atelectasis': 0.21819314062595363,
  'Cardiomegaly': 0.16013783663511272,
  'Consolidation': 0.1485404103994369,
  'Edema': 0.09559662565588946,
  'Effusion': 0.30975477695465087,
  'Emphysema': 0.18105267882347104,
  'Fibrosis': 0.08295522779226297,
  'Hernia': 0.5549078226089478,
  'Infiltration': 0.2306660830974579,
  'Mass': 0.2510164678096771,
  'Nodule': 0.12614848017692562,
  'Pleural_Thickening': 0.12851620316505424,
  'Pneumonia': 0.06241033226251595,
  'Pneumothorax': 0.17654986083507535
}

