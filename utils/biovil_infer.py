from  typing import List
from typing import Tuple

import tempfile
from pathlib import Path

import torch
from IPython.display import display
from IPython.display import Markdown

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine
from health_multimodal.image.model.encoder import ImageEncoder

from pathlib import Path
img = Path("coding/constrastive_P/diagnosisP/exchange/Fine-Grained_Features_Alignment_via_Constrastive_Learning/imgs/AUCcomp.png")
image_inference = get_image_inference(ImageModelType.BIOVIL_T)
embedding = image_inference.get_projected_global_embedding(img)
print(embedding.shape)
print("complete!!!")

# the shape of image encoder output is [1 * 128]
