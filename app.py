import os
import requests
import inspect
import pandas as pd

DEFAULT_API_URL = "https://jofthomas-unit4-scoring.hf.space/"

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

    def __repr__(self) -> str:
        imports = ["import inspect\n"]
        class_source = inspect.getsource(BasicAgent)
        full_source = "\n".join(imports) + "\n" + class_source
        return full_source

def get_current_script_content() -> str:
    try:
        script_path = os.path.abspath(__file__)
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Cannot read script content: {e}")
        return "# Agent code unavailable"

def run_and_submit_all(username: str):
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
        agent_code = get_current_script_content()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return

    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({
                "task_id": task_id,
                "submitted_answer": submitted_answer
            })
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
            print(f"Agent error on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}"
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }

    print(f"Submitting answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=45)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"\nSubmission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score')}% "
            f"({result_data.get('correct_count')}/{result_data.get('total_attempted')})\n"
            f"Message: {result_data.get('message')}"
        )
        print(final_status)
    except Exception as e:
        print(f"Submission failed: {e}")

    # Display table of results
    results_df = pd.DataFrame(results_log)
    print("\n--- Results Table ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    print("=== Basic Agent Evaluation CLI ===")
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("Username is required to proceed.")
    else:
        run_and_submit_all(username)
