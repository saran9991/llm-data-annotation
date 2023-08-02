import openai
from tenacity import (
retry,
stop_after_attempt,
wait_random_exponential,
)

with open('openai/organization.txt', 'r') as file:
    openai.organization = file.read().strip()

with open('openai/key.txt', 'r') as file:
    openai.api_key = file.read().strip()

accumulated_tokens = 0
accumulated_cost = 0
cost_per_token = 0.0035 / 1000  # The total cost per token, input and output
index = 0

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def analyze_gpt35(text):
    global index
    global accumulated_cost
    global accumulated_tokens
    messages = [
        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given text. Here are some examples:
                                            User: 'I love this!'; You: 'positive'
                                            User: 'I hate this!'; You: 'negative'
                                            User: 'I don't feel strongly about this.'; You: 'neutral' """},
        {"role": "user", "content": f"""Analyze the following texts and determine if the sentiment is: positive, negative, or neutral.
                                        Return answer in a single word as either 'positive', 'negative', or 'neutral': '{text}'"""}
        ]

    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      max_tokens=10,
                      n=1,
                      stop=None,
                      temperature=0)

    total_tokens_used = response['usage']['total_tokens']
    print(f"Total tokens used for this call: {total_tokens_used}")

    call_cost = total_tokens_used * cost_per_token
    accumulated_cost += call_cost
    accumulated_tokens += total_tokens_used
    index+=1
    print('Index: ', index)
    print(f"Cost for this call: {call_cost}")
    print(f"Accumulated tokens so far: {accumulated_tokens}")
    print(f"Accumulated cost so far: {accumulated_cost}\n")

    response_text = response.choices[0].message.content.strip().lower()

    return response_text