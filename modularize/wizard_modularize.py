from vllm import LLM, SamplingParams
import os
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset, Dataset
import json
import os
import numpy as np
import torch
import pickle
from examples import *
from openai import OpenAI
from torch.nn import CosineSimilarity
import torch
from utils import *
import argparse

client =OpenAI(your_key)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def batch(iterable, batch_size=256):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def get_text(output):
    return output.outputs[0].text


def module_prompt(problem):
    try:
        in_out = json.loads(problem["input_output"])
    except Exception as e:
        in_out = None
    if in_out is not None and in_out.get("fn_name") is not None:
        fn_name = in_out.get("fn_name")
        prompt = (
            WIZARD_MODULARIZE_PROMPT1
            + modularize_start.format(fn_name=fn_name)
            + WIZARD_MODULARIZE_PROMPT2.format(
                example_question=EXAMPLE_QUESTION,
                example_module1=EXAMPLE_MODULE1,
                example_module2=EXAMPLE_MODULE2,
                question=problem["question"],
                starter_code=problem["starter_code"],
            )
        )
    else:
        fn_name = None
        prompt = WIZARD_MODULARIZE_PROMPT1 + WIZARD_MODULARIZE_PROMPT2.format(
            example_question=EXAMPLE_QUESTION,
            example_module1=EXAMPLE_MODULE1,
            example_module2=EXAMPLE_MODULE2,
            question=problem["question"],
            starter_code=problem["starter_code"],
        )
    return prompt


def main(args):

    if args.model == "codellama":
        llm = LLM(
            model="codellama/CodeLlama-13b-hf",
            trust_remote_code=True,
            dtype=torch.float16,
            max_model_len=8192,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
        )
    elif args.model == "wizard":
        llm = LLM(
            model="WizardLM/WizardCoder-Python-13B-V1.0",
            trust_remote_code=True,
            dtype=torch.float16,
            max_model_len=8192,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
        )
    apps = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    if args.level == "all":
        data = apps
    else:
        data = apps.filter(lambda x: x["difficulty"] == args.level)

    prompts = list(map(module_prompt, data))
    data = data.add_column(name="prompt", column=prompts)
    # inputs = batch(prompts)

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=1024,
        stop=["\n\n\n\n", "RESPONSE:", "TASK", "###", "import"],
    )

    outputs = llm.generate(data["prompt"], sampling_params)
    output = list(map(get_text, outputs))
    data = data.add_column(name="modular_out", column=output)
    # modules = list(map(get_module, output))
    # data = data.add_column(name="modules", column=modules)

    # answers = []
    # similarity = []
    # cos = CosineSimilarity(dim=-1, eps=1e-6)
    # for input in tqdm(data):
    #     embed_question = get_embedding(input["question"])
    #     best_sim = -100
    #     best_module = None
    #     for i, module in enumerate(input["modules"]):
    #         try:
    #             embed_module = get_embedding(module)
    #             sim_module = cos(
    #                 torch.tensor(embed_question), torch.tensor(embed_module)
    #             )
    #             if sim_module > best_sim:
    #                 best_sim = sim_module
    #                 best_module = module
    #             # if best_sim < 0.5:
    #             #     다시 생성하게끔?
    #         except Exception:
    #             pass
    #     answers.append(best_module)
    #     similarity.append(best_sim)

    # data = data.add_column(name="best_module", column=answers)
    # data = data.add_column(
    #     name="similarity", column=list(torch.tensor(similarity).numpy())
    # )
    # # if os.path.exists(f"file/for_generation_{args.level}_wizard.parquet"):
    # #     data = Dataset.from_parquet(f"file/for_generation_{args.level}_wizard.parquet")
    # data = data.add_column(
    #     name="final_prompt",
    #     column=[final_wizard_prompt(da) for da in data],
    # )
    data.to_parquet(f"file/for_generation_{args.level}_wizard.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="-----Modularization on APPS Dataset-----"
    )
    parser.add_argument("--model", type=str, default="wizard", help="Name of the model")
    parser.add_argument("--level", type=str, default="introductory", help="Difficulty")

    args = parser.parse_args()

    main(args)
