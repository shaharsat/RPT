import torch
import sys
sys.path.append('/Users/shahar.satamkar/Desktop/Masters/RPT')
sys.path.append('/a/home/cc/students/cs/ohadr/mteb/RPT')
print(sys.path)
from mteb import AmazonReviewsClassification, Banking77Classification
from EasyLM.models.neox.neox_model_torch import GPTNeoXForCausalLM
import mteb


"""
hf_model = GPTNeoXForCausalLM.from_pretrained(
    pretrained_model_name_or_path='/Users/shahar.satamkar/Desktop/Masters/RPT/EasyLM/models/neox/neox_model_torch'
)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'neox_rpt_model'
hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1')
hf_model.to(device=device)

evaluation = mteb.MTEB(tasks=[Banking77Classification(hf_subsets=["en"], batch_size=16)])
results = evaluation.run(hf_model, output_folder=f"results/{model_name}")

