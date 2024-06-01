import sys

import torch.nn
from mteb import AmazonReviewsClassification, BUCCBitextMining

from EasyLM.models.neox.neox_model_torch import GPTNeoXForCausalLM
from EasyLM.models.neox.neox_serve_torch import load_model
from sklearn.utils import Bunch
import json
import yaml
import mteb

model_name = 'rpt'

def get_config():
    with open("tmpc8l0ys80.json") as f:
        config = json.load(f)

    config = Bunch(**config['mlxu.config'])
    config.config_override = ["apply_refactor=True"]
    config.append_next_chunk=True
    return config

config = get_config()
config['input_length'] = 2048

hf_model = GPTNeoXForCausalLM.from_pretrained(pretrained_model_name_or_path='/Users/shahar.satamkar/Desktop/Masters/RPT/EasyLM/models/neox/neox_model_torch')

evaluation = mteb.MTEB(tasks=[AmazonReviewsClassification(hf_subsets=["en"], batch_size=2)])
results = evaluation.run(hf_model, output_folder=f"results/{model_name}")

