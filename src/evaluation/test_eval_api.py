from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

model = HFLM(pretrained="gpt2")

results = simple_evaluate(
    model=model,
    tasks="arc_easy",
    num_fewshot=0,
    batch_size="auto:1"
)

print(f"Results on ARC-easy: {results['results']}")