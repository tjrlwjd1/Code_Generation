import re
import json
import torch
import random
import numpy as np
from examples import *


def get_module(answer):
    pattern = r"def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\("
    modules = re.findall(pattern, answer)
    return modules


def get_solution(answer):
    pattern = r"\[PYTHON\]([\s\S]*?)\[/PYTHON\]"
    modules = re.findall(pattern, answer)
    return modules


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def batch(iterable, batch_size=256):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def final_prompt(dataset):
    in_out = json.loads(dataset["input_output"])
    if "fn_name" in in_out:
        fn_name = in_out["fn_name"]
    else:
        fn_name = None
    if fn_name is not None:
        return (
            SOLUTION_PROMPT1
            + solution_start.format(fn_name=fn_name)
            + SOLUTION_PROMPT2.format(
                example_question=EXAMPLE_QUESTION,
                example_module1=EXAMPLE_MODULE1,
                example_module2=EXAMPLE_MODULE2,
                final_solution=EXAMPLE_SOLUTION,
                question=dataset["question"],
                starter_code=dataset["starter_code"],
            )
        )
    else:
        return SOLUTION_PROMPT1 + SOLUTION_PROMPT2.format(
            example_question=EXAMPLE_QUESTION,
            example_module1=EXAMPLE_MODULE1,
            example_module2=EXAMPLE_MODULE2,
            final_solution=EXAMPLE_SOLUTION,
            question=dataset["question"],
            starter_code=dataset["starter_code"],
        )


def final_wizard_prompt(dataset):
    in_out = json.loads(dataset["input_output"])
    if "fn_name" in in_out:
        fn_name = in_out["fn_name"]
    else:
        fn_name = None
    if fn_name is not None:
        return (
            WIZARD_PROMPT1
            + wizard_start.format(fn_name=fn_name)
            + WIZARD_PROMPT2.format(
                example_question=EXAMPLE_QUESTION,
                example_module1=EXAMPLE_MODULE1,
                example_module2=EXAMPLE_MODULE2,
                final_solution=EXAMPLE_SOLUTION,
                question=dataset["question"],
                module=dataset["best_module"],
                starter_code=dataset["starter_code"],
            )
        )
    else:
        return WIZARD_PROMPT1 + WIZARD_PROMPT2.format(
            example_question=EXAMPLE_QUESTION,
            example_module1=EXAMPLE_MODULE1,
            example_module2=EXAMPLE_MODULE2,
            final_solution=EXAMPLE_SOLUTION,
            question=dataset["question"],
            module=dataset["best_module"],
            starter_code=dataset["starter_code"],
        )
