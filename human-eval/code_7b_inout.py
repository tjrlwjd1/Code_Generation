from vllm import LLM, SamplingParams
import torch
import os
import tqdm
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness
from human_eval.evaluation import estimate_pass_at_k, evaluate_functional_correctness

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

problems = read_problems()
STOP_SEQS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=512,
#     skip_special_tokens=True,
#     stop=STOP_SEQS,
# )  # sampling

# llm = LLM(
#     model="codellama/CodeLlama-7b-hf",
#     trust_remote_code=True,
#     dtype=torch.float16,
#     tensor_parallel_size=2,
# )

# records = []
# for task_id in tqdm.tqdm(problems):
#     prompts_expand = [problems[task_id]["prompt"] for _ in range(200)]
#     outputs = llm.generate(prompts_expand, sampling_params=sampling_params)
#     for output in outputs:
#         completion = output.outputs[0].text
#         record = {"task_id": task_id, "completion": completion}
#         records.append(record)

# write_jsonl("codellama_vllm_200.jsonl", records)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# pass_at_k = evaluate_functional_correctness(
#     "file/codellama_vllm_200.jsonl",
#     k=[10, 100],
#     n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
# )

# print(pass_at_k)


# Below code is for Pass@1

sampling_params = SamplingParams(
    temperature=0, max_tokens=512, skip_special_tokens=True, stop=STOP_SEQS
)  # greedy decoding

llm = LLM(
    model="codellama/CodeLlama-7b-hf",
    trust_remote_code=True,
    dtype=torch.float16,
    gpu_memory_utilization=1,
)


def make_prompt(text):
    # return text
    return f'''def get_unique_elements(input_list: List[float]) -> List[float]:\n    """Get the unique elements of a list.\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4])\n    [1, 2, 3, 4]\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4, 4, 4, 4])\n    [1, 2, 3, 4]\n    """\n[PYTHON]\n    #input: [1,1,2,3,2], output: {{1,2,3}}\n    unique_set = set(my_list)\n    #input: {{1,2,3}}, output: [1,2,3]\n    unique_list = list(unique_set)\n\n    return unique_list\n[/PYTHON]\n[INST]\n{text}[/INST]'''


# def make_prompt(text):
#     # return text
#     return f'''def get_unique_elements(input_list: List[float]) -> List[float]:\n    """Get the unique elements of a list.\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4])\n    [1, 2, 3, 4]\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4, 4, 4, 4])\n    [1, 2, 3, 4]\n    """\n[PYTHON]\n    # Use the 'set' function to convert 'my_list' into a set, removing duplicate elements\n    unique_set = set(my_list)\n\n    \
# Convert the set back to a list to retain the order, creating a list of unique elements\n    unique_list = list(unique_set)\n\n    \
# Return the resulting list containing unique elements\n    return unique_list\n[/PYTHON]\n\
# [INST]\n{text}[/INST]'''


inputs = [problems[task_id]["prompt"] for task_id in problems]
input = list(map(make_prompt, inputs))

records = []
outputs = llm.generate(input, sampling_params=sampling_params)

answers = []
for output in outputs:
    gen = output.outputs[0].text
    answers.append(
        [gen[gen.find("[PYTHON]") + len("[PYTHON]") : gen.find("[/PYTHON]")]]
    )
    # answers.append([gen])

for i, task_id in tqdm.tqdm(enumerate(problems)):
    completion = answers[i][0]
    record = {"task_id": task_id, "completion": completion}
    records.append(record)

write_jsonl("file/codellama_vllm_greedy_inout.jsonl", records)
pass_at_1 = evaluate_functional_correctness(
    "file/codellama_vllm_greedy_inout.jsonl",
    k=[1],
    n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
)

print(pass_at_1)
