import dspy
from dspy import OllamaLocal
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# -----------------------------------
# LLM
# -----------------------------------

# model
model = OllamaLocal(model='llama2')
dspy.configure(lm=model)

# test - call model directly
result = model("Tell me a joke.")
print(result)

# -----------------------------------
# Dataset
# -----------------------------------
# Load math questions from the GSM8K dataset
gms8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gms8k.train[:10], gms8k.dev[:10]


# -----------------------------------
# Modul
# -----------------------------------
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
    
# -----------------------------------
# Compile and Evaluate the Model
# -----------------------------------

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gms8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

# -----------------------------------
# Evaluate
# -----------------------------------

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

# -----------------------------------
# Inspect the Model's History
# -----------------------------------
model.inspect_history(n=1)
