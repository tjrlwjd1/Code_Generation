import re
import json
import torch
import random
import numpy as np
from examples import *
import gzip
import os


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def write_jsonl(filename, data, append=False):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        output = json.load(json_file)
    return output


def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


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
