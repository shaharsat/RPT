import torch
from mteb import MTEB

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM

model = RPTForCausalLM.from_pretrained('shahar603/rpt-torch-1')

device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

model.to(device)
#model.push_to_hub(repo_id="rpt-torch-1", token='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')

#model.encode(['hello world', 'this is a test'])

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/'rpt")

