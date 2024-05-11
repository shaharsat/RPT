import torch
from mteb import MTEB
from torch import nn

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM

model = RPTForCausalLM.from_pretrained('shahar603/rpt-torch-1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/'rpt")

