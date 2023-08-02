import openai

with open('openai/organization.txt', 'r') as file:
    openai.organization = file.read().strip()

with open('openai/key.txt', 'r') as file:
    openai.api_key = file.read().strip()

accumulated_tokens = 0
accumulated_cost = 0
cost_per_token = 0.0035 / 1000

def analyze_davinci(text):
    global accumulated_tokens
    global accumulated_cost

    prompt = f"Sentiment analysis for the following text in a single word: positive, neutral, negative: \"{text}\""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )

    total_tokens_used = response['usage']['total_tokens']
    print(f"Total tokens used for this call: {total_tokens_used}")

    call_cost = total_tokens_used * cost_per_token
    accumulated_cost += call_cost
    accumulated_tokens += total_tokens_used
    print(f"Cost for this call: {call_cost}")
    print(f"Accumulated tokens so far: {accumulated_tokens}")
    print(f"Accumulated cost so far: {accumulated_cost}\n")

    response_text = response.choices[0].text.strip().lower()

    return response_text
