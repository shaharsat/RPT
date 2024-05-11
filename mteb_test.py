import torch
from mteb import MTEB

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM

model = RPTForCausalLM.from_pretrained('shahar603/rpt-torch-1')

import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

model.to('cuda')
#model.push_to_hub(repo_id="rpt-torch-1", token='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')

#model.encode(['hello world', 'this is a test'])

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/'rpt")

