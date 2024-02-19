from datasets import load_dataset, Dataset
from eval.apps_metric import apps_metric
import pickle
import argparse


def main(args):
    apps = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    data = Dataset.from_parquet(
        f"file/data_out_{args.level}_{args.num_prob}_{args.num_sol}_wizard.parquet"
    )

    eval_apps = apps_metric()

    if args.level == "all":
        pass
    else:
        apps = apps.filter(lambda x: x["difficulty"] == args.level)

    generation = [[]] * len(apps)
    for i, prob in enumerate(data):
        generation[i] = prob["answers"]

    results, metrics = eval_apps._compute(generation, k_list=[1], level=args.level)

    pickle.dump(results, open(f"file/results_{args.level}_wizard_1.pkl", "wb"))
    pickle.dump(metrics, open(f"file/metrics_{args.level}_wizard_1.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----Evaluation on APPS Dataset-----")
    parser.add_argument("--level", type=str, default="introductory", help="Difficulty")
    parser.add_argument(
        "--num_sol", type=int, default=20, help="Number of Solutions per problem"
    )
    parser.add_argument("--num_prob", type=int, default=1000, help="Number of Problems")
    args = parser.parse_args()

    main(args)
