from mteb import AmazonReviewsClassification, Banking77Classification
from EasyLM.models.neox.neox_model_torch import GPTNeoXForCausalLM
import mteb


"""
hf_model = GPTNeoXForCausalLM.from_pretrained(
    pretrained_model_name_or_path='/Users/shahar.satamkar/Desktop/Masters/RPT/EasyLM/models/neox/neox_model_torch'
)
"""

model_name = 'neox_rpt_model'

hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1')

evaluation = mteb.MTEB(tasks=[Banking77Classification(hf_subsets=["en"], batch_size=2)])
results = evaluation.run(hf_model, output_folder=f"results/{model_name}")

