from utils import *
from vllm import LLM, SamplingParams
import torch
from datasets import Dataset
from tqdm import tqdm
import warnings
import argparse


def main(args):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    data = Dataset.from_parquet(f"file/for_generation_{args.level}_wizard.parquet")

    # inputs = batch(data["final_prompt"])
    if args.num_sol == 1:
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
            stop=["\n\n\n\n", "RESPONSE:", "TASK", "###"],
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            stop=["\n\n\n\n", "RESPONSE:", "TASK", "###"],
        )
    if args.model == "wizard":
        llm = LLM(
            model="WizardLM/WizardCoder-Python-13B-V1.0",
            trust_remote_code=True,
            dtype=torch.float16,
            max_model_len=8192,
            tensor_parallel_size=2,
        )
    elif args.model == "codellama":
        llm = LLM(
            model="codellama/CodeLlama-13b-hf",
            trust_remote_code=True,
            dtype=torch.float16,
            max_model_len=8192,
            tensor_parallel_size=2,
        )

    answers = []
    solutions_20 = []
    for i in tqdm(range(args.num_sol)):
        outputs = llm.generate(data["final_prompt"], sampling_params)
        for output in outputs:
            gen = output.outputs[0].text
            answers.append(gen)
        solutions = list(map(get_solution, answers))
        solutions_20.append(solutions)

    final_solutions = []
    for i in range(args.num_prob, 0, -1):
        final_solution = []
        for j in range(args.num_sol):
            if solutions_20[j][-i] != []:
                final_solution.append(solutions_20[j][-i][0])
            else:
                final_solution.append("")
        final_solutions.append(final_solution)

    data = data.add_column(name="answers", column=final_solutions)

    data.to_parquet(
        f"file/data_out_{args.level}_{args.num_prob}_{args.num_sol}_wizard.parquet"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----Generation on APPS Dataset-----")
    parser.add_argument("--model", type=str, default="wizard", help="Name of the model")
    parser.add_argument(
        "--num_sol", type=int, default=20, help="Number of Solutions per problem"
    )
    parser.add_argument("--num_prob", type=int, default=1000, help="Number of Problems")
    parser.add_argument("--level", type=str, default="introductory", help="Difficulty")
    args = parser.parse_args()

    main(args)
