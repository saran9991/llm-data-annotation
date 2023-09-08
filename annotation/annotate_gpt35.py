import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Tuple, List

def read_api_information(file_path: str) -> str:
    """
    Read and return API information from a given file.

    Parameters:
    - file_path (str): Path to the file containing API information.

    Returns:
    - str: API information read from the file.
    """
    with open(file_path, 'r') as file:
        return file.read().strip()

openai.organization = read_api_information('openai/organization.txt')
openai.api_key = read_api_information('openai/key.txt')

accumulated_tokens = 0
accumulated_cost = 0
cost_per_token = 0.0035 / 1000
index = 0

def calculate_cost(total_tokens_used: int) -> Tuple[float, List[str]]:
    """
    Calculate the cost and return logs.

    Parameters:
    - total_tokens_used (int): Total tokens used for an API call.

    Returns:
    - Tuple[float, List[str]]: A tuple containing the call cost and a list of logs.
    """
    global accumulated_tokens, accumulated_cost, index

    call_cost = total_tokens_used * cost_per_token
    accumulated_cost += call_cost
    accumulated_tokens += total_tokens_used
    index += 1

    logs = [
        f"Total tokens used for this call: {total_tokens_used}",
        f"Index: {index}",
        f"Cost for this call: {call_cost}",
        f"Accumulated tokens so far: {accumulated_tokens}",
        f"Accumulated cost so far: {accumulated_cost}\n"
    ]
    for log in logs:
        print(log)

    return call_cost, logs

@retry(wait=wait_random_exponential(max=2), stop=stop_after_attempt(2))
def analyze_gpt35(text: str) -> Tuple[str, float, List[str]]:
    """
    Analyze text and classify its sentiment using GPT-3.5.

    Parameters:
    - text (str): The text to be analyzed for sentiment.

    Returns:
    - Tuple[str, float, List[str]]: A tuple containing the primary sentiment classification,
      confidence score, and a list of logs.
    """
    messages = [
        {"role": "system", "content": "Your task is to analyze text and classify its sentiment as either 'positive', 'negative', or 'neutral' in a single word."},
        {"role": "user", "content": f"Classify the sentiment of: '{text}'."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=3,
        n=3,
        temperature=0.5
    )

    total_tokens_used = response['usage']['total_tokens']
    _, logs = calculate_cost(total_tokens_used)

    response_texts = [choice.message.content.strip().lower() for choice in response.choices]
    primary_response = response_texts[0]
    confidence_score = response_texts.count(primary_response) / 3

    return primary_response, confidence_score, logs
