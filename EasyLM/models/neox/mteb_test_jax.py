import jax
import sys
sys.path.append('/Users/shahar.satamkar/Desktop/Masters/RPT')
sys.path.append('/a/home/cc/students/cs/ohadr/mteb/RPT')
print(sys.path)
from mteb import AmazonReviewsClassification, Banking77Classification
from EasyLM.models.neox.neox_model import FlaxGPTNeoXForCausalLM
import mteb


model_name = 'neox_rpt_model'
hf_model = FlaxGPTNeoXForCausalLM.from_pretrained('iohadrubin/rpt-2-1.6b_7529ebf46738fdc75e44')

evaluation = mteb.MTEB(tasks=[Banking77Classification(hf_subsets=["en"], batch_size=16)])
results = evaluation.run(hf_model, output_folder=f"results/{model_name}")

