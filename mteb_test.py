import torch
from mteb import MTEB
from torch import nn

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM

model = RPTForCausalLM.from_pretrained('shahar603/rpt-torch-1')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

#model.push_to_hub(repo_id="rpt-torch-1", token='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')

#model.encode(['hello world', 'this is a test'])

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/rpt", batch_size=2, show_progress_bar=True)

