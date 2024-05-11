import torch
from mteb import MTEB

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM
model = RPTForCausalLM.from_pretrained('shahar603/rpt-torch-1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
#model.encode(['hello world', 'this is a test'])

#rom sentence_transformers import SentenceTransformer

#model_name = "average_word_embeddings_komninos"
#model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=["Banking77Classification"], n_experiments=1)
results = evaluation.run(model, output_folder=f"results/rpt", batch_size=3, show_progress_bar=True)




