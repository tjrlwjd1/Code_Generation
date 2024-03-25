from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import random
from utils import *
from tqdm import tqdm

options = webdriver.ChromeOptions()
options.add_argument("headless")

driver = webdriver.Chrome(options=options)

driver.get("https://github.com/tony9402/baekjoon/tree/main/solution")

codes = []
topics = driver.find_element(By.TAG_NAME, "tbody").find_elements(
    By.CLASS_NAME, "react-directory-row-name-cell-large-screen"
)
for j in tqdm(range(len(topics))):
    topics = driver.find_element(By.TAG_NAME, "tbody").find_elements(
        By.CLASS_NAME, "react-directory-row-name-cell-large-screen"
    )
    top = topics[j].find_element(By.TAG_NAME, "a").text
    topics[j].find_element(By.TAG_NAME, "a").click()
    time.sleep(1)
    problems = driver.find_element(By.TAG_NAME, "tbody").find_elements(
        By.CLASS_NAME, "react-directory-row-name-cell-large-screen"
    )
    idx = len(problems)

    for i in tqdm(range(idx)):
        problems = driver.find_element(By.TAG_NAME, "tbody").find_elements(
            By.CLASS_NAME, "react-directory-row-name-cell-large-screen"
        )

        prob_num = problems[i].find_element(By.TAG_NAME, "a").text
        problems[i].find_element(By.TAG_NAME, "a").click()
        time.sleep(1)
        try:
            driver.get(driver.current_url + "/main.py")
            time.sleep(1)

            code = driver.find_element(
                By.XPATH, '//*[@id="read-only-cursor-text-area"]'
            ).text

            codes.append({"topic": top, "problem": prob_num, "code": code})
            driver.back()
            driver.back()
            time.sleep(1)
        except:
            driver.back()
            driver.back()
            time.sleep(1)
    driver.back()
    time.sleep(1)
write_jsonl("codes.jsonl", codes)
