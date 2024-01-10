from vllm import LLM, SamplingParams
import torch
import os
import tqdm
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness
from human_eval.evaluation import estimate_pass_at_k, evaluate_functional_correctness

os.environ["TOCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

problems = read_problems()
STOP_SEQS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
    skip_special_tokens=True,
    stop=STOP_SEQS,
)  # sampling

llm = LLM(
    model="codellama/CodeLlama-7b-hf",
    trust_remote_code=True,
    dtype=torch.float16,
    tensor_parallel_size=2,
)

records = []
for task_id in tqdm.tqdm(problems):
    prompts_expand = [problems[task_id]["prompt"] for _ in range(200)]
    outputs = llm.generate(prompts_expand, sampling_params=sampling_params)
    for output in outputs:
        completion = output.outputs[0].text
        record = {"task_id": task_id, "completion": completion}
        records.append(record)

write_jsonl("codellama_vllm_200.jsonl", records)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pass_at_k = evaluate_functional_correctness(
    "codellama_vllm_200.jsonl",
    k=[10, 100],
    n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
)

print(pass_at_k)


# Below code is for Pass@1

sampling_params = SamplingParams(
    temperature=0, max_tokens=512, skip_special_tokens=True, stop=STOP_SEQS
)  # greedy decoding

llm = LLM(
    model="codellama/CodeLlama-7b-hf",
    trust_remote_code=True,
    dtype=torch.float16,
    tensor_parallel_size=2,
)

records = []
for task_id in tqdm.tqdm(problems):
    outputs = llm.generate(problems[task_id]["prompt"], sampling_params=sampling_params)
    for output in outputs:
        completion = output.outputs[0].text
        record = {"task_id": task_id, "completion": completion}
        records.append(record)

write_jsonl("codellama_vllm_greedy.jsonl", records)
pass_at_1 = evaluate_functional_correctness(
    "codellama_vllm_greedy.jsonl",
    k=[1],
    n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
)

print(pass_at_1)
