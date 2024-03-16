from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset
import json
import torch
import pickle
import re
from datasets import Dataset
from vllm import LLM, SamplingParams
import random
from eval.apps_metric import apps_metric
from openai import OpenAI
from tqdm import tqdm


def extract_answer(text):
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    if "```python" not in text:
        pattern = re.compile(r"```\n(.*?)\n```", re.DOTALL)
    match = pattern.search(text)[1]
    return match


apps = load_dataset("codeparrot/apps", trust_remote_code=True, split="test")
apps_train = load_dataset("codeparrot/apps", trust_remote_code=True, split="train")

apps_introductory = apps.filter(lambda x: x["difficulty"] == "introductory")
apps_train_introductory = apps_train.filter(lambda x: x["difficulty"] == "introductory")

client = OpenAI(api_key=your_key)


def make_modular(problem):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to create a modular Python code based on the below Python code. You need to generate code that is different from the below Python code. Please wrap your code with ```python and ```.",
                },
                {
                    "role": "system",
                    "name": "example_user",
                    "content": """
import sys

class GCandyBoxHardVersion:
    def solve(self):
        q = int(input())
        for _ in range(q):
            n = int(input())
            a = [0] * n
            f = [0] * n
            for i in range(n):
                a[i], f[i] = [int(_) for _ in input().split()]

            d = {key: [0, 0] for key in a}
            for i in range(n):
                d[a[i]][f[i]] += 1
            rev_d = {sum(key): [] for key in list(d.values())}
            for x in d:
                rev_d[d[x][0] + d[x][1]] += [d[x]]

            for x in rev_d:
                rev_d[x].sort(key=lambda item:item[1])

            # print(rev_d)

            cur = max(rev_d)
            cnt = max(rev_d)
            nb_candies = 0
            given_away = 0
            while 1:
                if cnt == 0 or cur == 0:
                    break
                if cur > cnt:
                    cur -= 1
                    continue

                if cnt not in rev_d or not rev_d[cnt]:
                    cnt -= 1
                    continue

                mx_f = -1
                v = -1
                for max_cnt in range(cur, cnt + 1):
                    if max_cnt in rev_d and rev_d[max_cnt] and rev_d[max_cnt][-1][1] > mx_f:
                        v = max_cnt
                        mx_f = rev_d[max_cnt][-1][1]
                to_take = rev_d[v].pop()
                # rev_d[cnt] -= 1
                nb_candies += cur
                given_away += min(to_take[1], cur)
                cur -= 1
                # rev_d[cnt - cur] += 1
            print(nb_candies, given_away)

solver = GCandyBoxHardVersion()
input = sys.stdin.readline

solver.solve()
""",
                },
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": """
import sys

class GCandyBoxHardVersion:
    def read_input(self):
        n = int(input())
        a, f = [0] * n, [0] * n
        for i in range(n):
            a[i], f[i] = map(int, input().split())
        return n, a, f

    def process_candies(self, n, a, f):
        d = {key: [0, 0] for key in a}
        for i in range(n):
            d[a[i]][f[i]] += 1
        rev_d = {sum(value): [] for value in d.values()}
        for x in d:
            rev_d[sum(d[x])].append(d[x])
        for x in rev_d:
            rev_d[x].sort(key=lambda item: item[1])
        return rev_d

    def calculate_gift(self, rev_d):
        cur = max(rev_d)
        cnt = cur
        nb_candies = 0
        given_away = 0
        while cnt > 0 and cur > 0:
            if cur > cnt:
                cur -= 1
                continue

            if cnt not in rev_d or not rev_d[cnt]:
                cnt -= 1
                continue

            mx_f = -1
            v = -1
            for max_cnt in range(cur, cnt + 1):
                if max_cnt in rev_d and rev_d[max_cnt] and rev_d[max_cnt][-1][1] > mx_f:
                    v = max_cnt
                    mx_f = rev_d[max_cnt][-1][1]
            to_take = rev_d[v].pop()
            nb_candies += cur
            given_away += min(to_take[1], cur)
            cur -= 1
        return nb_candies, given_away

    def solve(self):
        q = int(input())
        for _ in range(q):
            n, a, f = self.read_input()
            rev_d = self.process_candies(n, a, f)
            nb_candies, given_away = self.calculate_gift(rev_d)
            print(nb_candies, given_away)

solver = GCandyBoxHardVersion()
input = sys.stdin.readline
solver.solve()
""",
                },
                {"role": "user", "content": json.loads(problem["solutions"])[0]},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error ; {e}")
        return None


answer = list(map(make_modular, tqdm(apps_train_introductory)))
pickle.dump(answer, open("file/gpt_answer_introductory.pkl", "wb"))
