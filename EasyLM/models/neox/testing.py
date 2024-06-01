# DEBUG=1 python3.10 /home/ohadr/EasyLM/EasyLM/etc/test_serve.py
import sys

import torch.nn

#from EasyLM.nq_data import NQDataset
sys.path.append('/home/ohadr/EasyLM')
from EasyLM.models.neox.neox_serve_torch import load_model

import json
from sklearn.utils import Bunch


import json
import yaml
def get_decoded_body():
    file_path = "recorded_session.yaml"
    with open(file_path, "r") as f:
        session = yaml.safe_load(f)
    session = session["interactions"]
    encoded_body = session[-1]["request"]["body"]
    return json.loads(encoded_body)


def get_config():
    with open("tmpc8l0ys80.json") as f:
        config = json.load(f)


    config = Bunch(**config['mlxu.config'])
    config.config_override = ["apply_refactor=True"]
    config.append_next_chunk=True
    return config
config = get_config()
config['input_length'] = 128

_loglikelihood, _loglikelihood_rolling, _generate, _greedy_until, _encode, _old_greedy_until = load_model(config)



from more_itertools import chunked
"""
def get_iter():
    val_iterator = iter(NQDataset(split='validation',transform=True))
    for data in  chunked(val_iterator,16):
        yield {"prefix_text":data, "until":[['\n', '.', ','] for _ in range(16)]}
"""

decoded_body = get_decoded_body()
print("hi")
print(config)

#encode_output = _encode([''.join(['hello' for _ in range(64)] + ['world' for _ in range(64)])])
encode_output = _encode(['hello'*100]*100)
print(encode_output)

#out = _greedy_until(max_length=100,**decoded_body)
#for x in get_iter():
#    print(x)
#    print(_greedy_until(max_length=100, **x))
# return
