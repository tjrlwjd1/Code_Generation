__all__ = [
    "example_code_main",
    "EXAMPLE_QUESTION",
    "EXAMPLE_MODULE1",
    "EXAMPLE_MODULE2",
    "EXAMPLE_MODULE3",
    "EXAMPLE_SOLUTION",
    "MODULARIZE_PROMPT1",
    "modularize_start",
    "MODULARIZE_PROMPT2",
    "SOLUTION_PROMPT1",
    "solution_start",
    "SOLUTION_PROMPT2",
    "WIZARD_MODULARIZE_PROMPT1",
    "wizard_modularize_start",
    "WIZARD_MODULARIZE_PROMPT2",
    "WIZARD_PROMPT1",
    "wizard_start",
    "WIZARD_PROMPT2",
]


example_code_main = """
for _ in range(int(input())):
    n = int(input())
    mass = []
    zo = 0
    oz = 0
    zz = 0
    oo = 0
    ozs = []
    zos = []
    ozss = set()
    zoss = set()
    for j in range(n):
        k = input()
        mass.append(k)
        if k[0] == '0' and k[-1] == '1':
            zoss.add(k)
            zos.append(j + 1)
            zo += 1
        elif k[0] == '1' and k[-1] == '0':
            ozss.add(k)
            ozs.append(j + 1)
            oz += 1
        elif k[0] == '0' and k[-1] == '0':
            zz += 1
        else:
            oo += 1
    if zz and oo and not oz and not zo:
        print(-1)
        continue
    else:
        if zo > oz:
            print((zo - oz) // 2)
            ans = []
            need = (zo - oz) // 2
            i = 0
            while need:
                zzz = mass[zos[i] - 1][len(mass[zos[i] - 1]) - 1:: -1]
                if zzz not in ozss:
                    ans.append(zos[i])
                    need -= 1
                i += 1
            print(*ans)
        else:
            print((oz - zo) // 2)
            ans = []
            need = (oz - zo) // 2
            i = 0
            while need:
                zzz = mass[ozs[i] - 1][len(mass[ozs[i] - 1]) - 1:: -1]
                if zzz not in zoss:
                    ans.append(ozs[i])
                    need -= 1
                i += 1
            print(*ans)
"""


EXAMPLE_QUESTION = """Polycarp has $n$ different binary words. A word called binary if it contains only characters '0' and '1'. For example, these words are binary: "0001", "11", "0" and "0011100".

Polycarp wants to offer his set of $n$ binary words to play a game "words". In this game, players name words and each next word (starting from the second) must start with the last character of the previous word. The first word can be any. For example, these sequence of words can be named during the game: "0101", "1", "10", "00", "00001".

Word reversal is the operation of reversing the order of the characters. For example, the word "0111" after the reversal becomes "1110", the word "11010" after the reversal becomes "01011".

Probably, Polycarp has such a set of words that there is no way to put them in the order correspondent to the game rules. In this situation, he wants to reverse some words from his set so that:  the final set of $n$ words still contains different words (i.e. all words are unique);  there is a way to put all words of the final set of words in the order so that the final sequence of $n$ words is consistent with the game rules. 

Polycarp wants to reverse minimal number of words. Please, help him.


-----Input-----

The first line of the input contains one integer $t$ ($1 \le t \le 10^4$) — the number of test cases in the input. Then $t$ test cases follow.

The first line of a test case contains one integer $n$ ($1 \le n \le 2\cdot10^5$) — the number of words in the Polycarp's set. Next $n$ lines contain these words. All of $n$ words aren't empty and contains only characters '0' and '1'. The sum of word lengths doesn't exceed $4\cdot10^6$. All words are different.

Guaranteed, that the sum of $n$ for all test cases in the input doesn't exceed $2\cdot10^5$. Also, guaranteed that the sum of word lengths for all test cases in the input doesn't exceed $4\cdot10^6$.


-----Output-----

Print answer for all of $t$ test cases in the order they appear.

If there is no answer for the test case, print -1. Otherwise, the first line of the output should contain $k$ ($0 \le k \le n$) — the minimal number of words in the set which should be reversed. The second line of the output should contain $k$ distinct integers — the indexes of the words in the set which should be reversed. Words are numerated from $1$ to $n$ in the order they appear. If $k=0$ you can skip this line (or you can print an empty line). If there are many answers you can print any of them.


-----Example-----
Input
4
4
0001
1000
0011
0111
3
010
101
0
2
00000
00001
4
01
001
0001
00001

Output
1
3 
-1
0

2
1 2"""


EXAMPLE_MODULE1 = '''
def reverse_words(string):
    """
    Description: this function reverses each word in the given string.
    Input:
    string (str): the input string.
    Output:
    reversed_string (str): the reversed string with each word reversed.
    """
    return ' '.join(s[::-1] for s in string.split(' '))
'''


EXAMPLE_MODULE2 = '''
def count_start_end_chars(words):
    """
    Description: This function counts the number of words that start and end with each character.
    Input:
    words (list): A list of binary words.
    Output:
    start_count (defaultdict): A dictionary containing the count of words that start with each character.
    end_count (defaultdict): A dictionary containing the count of words that end with each character.
    """
    start_count = collections.defaultdict(int)
    end_count = collections.defaultdict(int)
    for word in words:
        start_count[word[0]] += 1
        end_count[word[-1]] += 1
    return start_count, end_count
'''


EXAMPLE_MODULE3 = '''
def solve_task(words):
    """
    Description: This function analyzes a list of words to determine how many and which words should be reversed to balance the count of characters that start and end words in the list.
    Input: words (list): A list of binary words.
    Output:
    total_reversed (int): An integer representing the total number of words reversed.
    reversed_words (list): The list of words after reversing the specified words.
    """
    start_count, end_count = count_start_end_chars(words)
    characters_with_difference = []
    for char in start_count:
        if abs(start_count[char] - end_count[char]) > 1:
            characters_with_difference.append(char)
    reversed_indices = []
    for char in characters_with_difference:
        difference = abs(start_count[char] - end_count[char])
        reverse_count = difference // 2
        if start_count[char] < end_count[char]:
            indices = [i for i, word in enumerate(words) if word.startswith(char)]
            reversed_indices.extend(indices[:reverse_count])
        else:
            indices = [i for i, word in enumerate(words) if word.endswith(char)]
            reversed_indices.extend(indices[:reverse_count])
    reversed_words = reverse_words(words, reversed_indices)
    total_reversed = len(reversed_indices)
    return total_reversed, reversed_words
'''


EXAMPLE_SOLUTION = """
import collections

t = int(input())
for _ in range(t):
    n = int(input())
    words = []
    for _ in range(n):
    words.append(input())
    start_count, end_count = count_start_end_chars(words)
    characters_with_difference = []
    for char in start_count:
        if abs(start_count[char] - end_count[char]) > 1:
            characters_with_difference.append(char)
    reversed_indices = []
    for char in characters_with_difference:
        difference = abs(start_count[char] - end_count[char])
        reverse_count = difference // 2
        if start_count[char] < end_count[char]:
            indices = [i for i, word in enumerate(words) if word.startswith(char)]
            reversed_indices.extend(indices[:reverse_count])
        else:
            indices = [i for i, word in enumerate(words) if word.endswith(char)]
            reversed_indices.extend(indices[:reverse_count])
    reversed_words = reverse_words(words, reversed_indices)
    total_reversed = len(reversed_indices)
    print(total_reversed)
    if total_reversed != 0:
        print(*reversed_words)
"""


MODULARIZE_PROMPT1 = """<s>[INST]<<SYS>>\nBefore developing a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases, outline the required code modules, including function headers, signatures and bodies.\
Ensure modularity and considering potential edge cases and failures.\n\nIn simpler terms, break it down into smaller parts (modules) with clear function names and input/output specifications.<</SYS>>

"""


modularize_start = """\nThe output code needs to "{fn_name}"."""


MODULARIZE_PROMPT2 = """
###Example 1
### TASK:
{example_question}
[/INST]
### RESPONSE:
[MODULE]
{example_module1}
[/MODULE]
[MODULE]
{example_module2}
[/MODULE]
[INST]
### TASK:
{question}

{starter_code}
[/INST]
### RESPONSE:
"""


SOLUTION_PROMPT1 = """<s>[INST]<<SYS>>\nDevelop a well-structured Python solution for the provided problem that obeys the constraints and passes the example test cases. Ensure modularity and considering potential edge cases and failures.\
Given a set of related utility Python functions between [MODULE] and [/MODULE], try to reuse them as much as possible into your solution (create new unique functions if needed).\

In simpler terms, create a clean and organized Python solution using given module for the given problem. Your code answer must be between [PYTHON] and [/PYTHON].<</SYS>>

"""


solution_start = """\nThe output code needs to "{fn_name}"."""


SOLUTION_PROMPT2 = """
###Example 1
### TASK:
{example_question}
[MODULE]
{example_module1}
[/MODULE]
[MODULE]
{example_module2}
[/MODULE]
[/INST]
### RESPONSE:
[PYTHON]
{final_solution}
[/PYTHON]
[INST]
### TASK:
{question}

{starter_code}
[/INST]
### RESPONSE:
"""


WIZARD_MODULARIZE_PROMPT1 = """Below is an instruction that describes a task, paired with an input that provides further context. \
Write a response that appropriately completes the request."""


wizard_modularize_start = """And the output code needs to "{fn_name}"."""


WIZARD_MODULARIZE_PROMPT2 = """
###Example
### Instruction:
Create the required code modules which mean whole functions for solving this problem:
{example_question}

### Response:
{example_module1}

{example_module2}

### Instruction:
{question}

{starter_code}
### Response:
"""


WIZARD_PROMPT1 = """Below is an instruction that describes a task, paired with an input that provides further context. \
Write a response that appropriately completes the request."""


wizard_start = """ And the output code should contain a function {fn_name}."""


WIZARD_PROMPT2 = """

### Example
### Instruction:
Create a Python script for this problem using given module without any changes: 
{example_question}

### Module:
{example_module1}

{example_module2}

### Response:
{final_solution}

### Instruction:
Create a Python script for this problem using given module without any changes: 
{question}

### Module:
{module}

{starter_code}
### Response:
"""
