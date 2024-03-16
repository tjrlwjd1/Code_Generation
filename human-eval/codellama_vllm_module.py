from vllm import LLM, SamplingParams
import torch
import os
import tqdm
from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness
from human_eval.evaluation import estimate_pass_at_k, evaluate_functional_correctness
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
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

# write_jsonl("file/codellama_vllm_200.jsonl", records)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# pass_at_k = evaluate_functional_correctness(
#     "file/codellama_vllm_200.jsonl",
#     k=[10, 100],
#     n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
# )

# print(pass_at_k)

# extract = pickle.load(open("extract.pkl", "rb"))
### Below code is for Pass@1

# sampling_params = SamplingParams(
#     temperature=0, max_tokens=512, skip_special_tokens=True, stop=STOP_SEQS
# )  # greedy decoding

# llm = LLM(
#     model="codellama/CodeLlama-7b-hf",
#     trust_remote_code=True,
#     dtype=torch.float16,
#     tensor_parallel_size=2,
# )


# def make_prompt(text, module):
#     prompt = f'''[INST] Your task is to write a Python function to solve a programming problem given modules. The Python code must be between [PYTHON] and [/PYTHON] tags.
#     Problem:
#     def even_odd_palindrome(n):
#     """
#     Given a positive integer n, return a tuple that has the number of even and odd
#     integer palindromes that fall within the range(1, n), inclusive.

#     Example 1:

#         Input: 3
#         Output: (1, 2)
#         Explanation:
#         Integer palindrome are 1, 2, 3. one of them is even, and two of them are odd.

#     Example 2:

#         Input: 12
#         Output: (4, 6)
#         Explanation:
#         Integer palindrome are 1, 2, 3, 4, 5, 6, 7, 8, 9, 11. four of them are even, and 6 of them are odd.

#     Note:
#         1. 1 <= n <= 10^3
#         2. returned tuple has the number of even and odd integer palindromes respectively.
#     """
#     Functions:
#     def is_palindrome(num):
#         """Check if a number is a palindrome."""
#         return str(num) == str(num)[::-1]

#     def count_even_odd_palindromes(n):
#         """Count even and odd palindromes up to n."""
#         even_count, odd_count = 0, 0

#         for num in range(1, n + 1):
#             if is_palindrome(num):
#                 if num % 2 == 0:
#                     even_count += 1
#                 else:
#                     odd_count += 1

#         return even_count, odd_count
#     [/INST]
#     [PYTHON]
#     def even_odd_palindrome(n):
#         def is_palindrome(num):
#             """Check if a number is a palindrome."""
#             return str(num) == str(num)[::-1]

#         def count_even_odd_palindromes(n):
#             """Count even and odd palindromes up to n."""
#             even_count, odd_count = 0, 0

#             for num in range(1, n + 1):
#                 if is_palindrome(num):
#                     if num % 2 == 0:
#                         even_count += 1
#                     else:
#                         odd_count += 1

#             return even_count, odd_count

#         return count_even_odd_palindromes(n)
#     [/PYTHON]
#     [INST]
#     Problem:
#     {text}
#     Functions:
#     {module}
#     [/INST]'''
#     return prompt


# def make_prompt(text):
#     return text  ###zero-shot


#     return f'''def get_unique_elements(input_list: List[float]) -> List[float]:\n    """Get the unique elements of a list.\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4])\n    [1, 2, 3, 4]\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4, 4, 4, 4])\n    [1, 2, 3, 4]\n    """\n[PYTHON]\n    #input: [1,1,2,3,2], output: {{1,2,3}}\n    unique_set = set(my_list)\n    #input: {{1,2,3}}, output: [1,2,3]\n    unique_list = list(unique_set)\n\n    return unique_list\n[/PYTHON]\n[INST]\n{text}[/INST]'''


# def make_prompt(text):
#     return f"""[INST] Your task is to write a Python function to solve a programming problem.
# The Python code must be between [PYTHON] and [/PYTHON] tags.
# Problem: {text}
# [/INST]"""


# def make_prompt(text):
#     # return text
#     return f'''[INST]\ndef get_unique_elements(input_list: List[float]) -> List[float]:\n    """Get the unique elements of a list.\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4])\n    [1, 2, 3, 4]\n    >>> get_unique_elements([1, 2, 3, 3, 2, 1, 4, 4, 4, 4])\n    [1, 2, 3, 4]\n    """\n[/INST]\n[PYTHON]\n    # Use the 'set' function to convert 'my_list' into a set, removing duplicate elements\n    unique_set = set(my_list)\n\n    \
# Convert the set back to a list to retain the order, creating a list of unique elements\n    unique_list = list(unique_set)\n\n    \
# Return the resulting list containing unique elements\n    return unique_list\n[/PYTHON]\n\
# [INST]\n{text}[/INST]'''


# inputs = [problems[task_id]["prompt"] for task_id in problems]

# input = list(map(make_prompt, inputs))
# input = [make_prompt(i, j) for i, j in zip(inputs, extract)]

# outputs = llm.generate(input, sampling_params=sampling_params)

# answers = []
# for output in outputs:
# gen = output.outputs[0].text
# answers.append([gen])
# answers.append(
#     [gen[gen.find("[PYTHON]") + len("[PYTHON]") : gen.find("[/PYTHON]")]]
# )
# pickle.dump(answers, open("file/answers_module.pkl", "wb"))

# records = []
# for i, task_id in tqdm.tqdm(enumerate(problems)):
#     completion = answers[i][0]
#     record = {"task_id": task_id, "completion": completion}
#     records.append(record)

module = pickle.load(open("file/modules.pkl", "rb"))
entry_points = [problems[task_id]["entry_point"] for task_id in problems]
task_id = [problems[task_id]["task_id"] for task_id in problems]
records = []
for id, ent, mod in zip(task_id, entry_points, module):
    record = {"task_id": id, "completion": "def " + mod[mod.find(ent) :]}
    records.append(record)

write_jsonl("file/modular.jsonl", records)
pass_at_1 = evaluate_functional_correctness(
    "file/modular.jsonl",
    k=[1],
    n_workers=8,  # In CodeLlama paper, pass@1 from greedy decoding
)

# print(pass_at_1)
