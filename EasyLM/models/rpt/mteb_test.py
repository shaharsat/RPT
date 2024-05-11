from mteb import MTEB
from sentence_transformers import SentenceTransformer

from EasyLM.models.rpt.rpt_model_torch import RPTForCausalLM

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

model = RPTForCausalLM.from_pretrained('rpt-torch-1')
#model.encode(['hello world', 'this is a test'])

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")

