import os
from dotenv import load_dotenv
import requests
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get values
HF_USERNAME = os.getenv("HF_USERNAME")
API_URL = os.getenv("API_URL", "https://jofthomas-unit4-scoring.hf.space/")  # Fallback to default

class BasicAgent:
    def __call__(self, question: str) -> str:
        return "This is a default answer."

def run_and_submit_all():
    if not HF_USERNAME:
        print("HF_USERNAME is not set in the environment.")
        return

    questions_url = f"{API_URL}/questions"
    submit_url = f"{API_URL}/submit"

    agent = BasicAgent()
    agent_code = inspect.getsource(BasicAgent)

    try:
        response = requests.get(questions_url, timeout=10)
        response.raise_for_status()
        questions = response.json()
    except Exception as e:
        print("Error fetching questions:", e)
        return

    answers = []
    for q in questions:
        answer = agent(q["question"])
        answers.append({
            "task_id": q["task_id"],
            "submitted_answer": answer
        })

    submission = {
        "username": HF_USERNAME,
        "agent_code": agent_code,
        "answers": answers
    }

    try:
        res = requests.post(submit_url, json=submission, timeout=15)
        res.raise_for_status()
        result = res.json()
        print("Submission Result:", result)
    except Exception as e:
        print("Submission failed:", e)

if __name__ == "__main__":
    run_and_submit_all()
