"""
Stage 2 JSON Instruction Dataset Generator
Uses UTSA's Llama 3.1 8B API as teacher model to generate structured JSON outputs.
Covers 5 required task types:
  1. JSON extraction from unstructured text
  2. Schema-constrained generation
  3. Exact-label classification with JSON output
  4. JSON repair / formatting correction
  5. Tool-call argument generation
"""

import json
import random
import time
import requests
from pathlib import Path

# ---------------------------------------------------------
# UTSA API Configuration
# ---------------------------------------------------------
API_KEY   = "utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem"
BASE_URL  = "http://149.165.173.247:8888/v1"
MODEL     = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_FILE     = "json_instruct_dataset.jsonl"
N_PER_TASK_TYPE = 30   # 30 x 5 = 150 total examples
MAX_RETRIES     = 3
SLEEP_BETWEEN   = 0.5  # seconds between API calls

# ---------------------------------------------------------
# Prompt Templates (one per task type)
# ---------------------------------------------------------

# --- Task 1: JSON Extraction ---
EXTRACTION_INSTRUCTIONS = [
    "Extract the person's name, age, and city from the following text and return as JSON.",
    "Extract all product names, prices, and quantities mentioned in the text into a JSON object.",
    "Parse the following email and extract sender, recipient, subject, and date as a JSON object.",
    "Extract the company name, job title, and salary range from the job posting as JSON.",
    "Extract all mentioned dates, locations, and event names from the text into JSON.",
]

EXTRACTION_INPUTS = [
    "John Smith, a 34-year-old software engineer from Austin, Texas, recently joined our team.",
    "We ordered 3 units of Widget A at $12.99 each and 5 units of Widget B at $7.50 each.",
    "From: alice@example.com To: bob@example.com Subject: Meeting Tomorrow Date: March 5, 2024",
    "Senior Data Scientist at TechCorp Inc. Salary: $120,000 - $150,000 per year.",
    "The Annual Tech Summit will be held on June 15, 2024 in San Francisco, followed by a workshop in New York on June 20.",
]

# --- Task 2: Schema-Constrained Generation ---
SCHEMA_INSTRUCTIONS = [
    'Generate a JSON object for a user profile matching this schema: {"name": string, "age": integer, "email": string, "is_active": boolean}',
    'Generate a JSON object for a product matching this schema: {"id": integer, "name": string, "price": float, "in_stock": boolean, "tags": list of strings}',
    'Generate a JSON object for a book matching this schema: {"title": string, "author": string, "year": integer, "genre": string, "rating": float}',
    'Generate a JSON object for a weather report matching this schema: {"city": string, "temperature_c": float, "humidity_percent": integer, "condition": string}',
    'Generate a JSON object for a restaurant matching this schema: {"name": string, "cuisine": string, "rating": float, "price_range": string, "open_now": boolean}',
]

SCHEMA_INPUTS = [
    "Create a profile for a fictional user named Maria who is 28 years old.",
    "Create a product entry for a blue wireless headphone priced at $79.99.",
    "Create an entry for the classic novel '1984' by George Orwell.",
    "Create a weather report for Tokyo on a rainy day.",
    "Create an entry for a highly-rated Italian restaurant that is currently open.",
]

# --- Task 3: Exact-label Classification ---
CLASSIFICATION_INSTRUCTIONS = [
    'Classify the sentiment of the following text. Return JSON with key "label" having one of these values: "positive", "negative", "neutral".',
    'Classify the topic of the following sentence. Return JSON with key "category" having one of: "technology", "sports", "politics", "entertainment", "science".',
    'Classify the urgency of the following support ticket. Return JSON with key "urgency" having one of: "low", "medium", "high", "critical".',
    'Classify the language of the following text. Return JSON with key "language" and key "confidence" (float 0-1).',
    'Classify whether the following news headline is clickbait. Return JSON with key "is_clickbait" (boolean) and key "reason" (string).',
]

CLASSIFICATION_INPUTS = [
    "I absolutely love this product! It exceeded all my expectations and works perfectly.",
    "Scientists have developed a new AI model that can predict protein structures with 99% accuracy.",
    "URGENT: Our production database is down and customers cannot place orders.",
    "Bonjour, comment puis-je vous aider aujourd'hui?",
    "You Won't Believe What This Celebrity Did at the Airport!",
]

# --- Task 4: JSON Repair ---
REPAIR_INSTRUCTIONS = [
    "Fix the malformed JSON below and return only the corrected valid JSON.",
    "The following JSON has syntax errors. Repair it and return valid JSON only.",
    "Correct all JSON formatting errors in the following text and return valid JSON.",
]

REPAIR_INPUTS = [
    '{name: "Alice", age: 30, city: "Boston"}',
    '{"product": "laptop", "price": 999.99, "in_stock": True, "tags": ["electronics", "computers"]}',
    '{"user": {"id": 1, "name": "Bob", "email": "bob@example.com", "roles": ["admin", "user"]}',
    "{'title': 'Python Guide', 'author': 'John', 'pages': 350}",
    '{"items": [{"id": 1, "name": "pen"}, {"id": 2, "name": "notebook",}], "total": 2}',
]

# --- Task 5: Tool-call Argument Generation ---
TOOL_INSTRUCTIONS = [
    'Generate a JSON object representing a function call to "search_web" with parameters: query (string), num_results (integer), safe_search (boolean).',
    'Generate a JSON object representing a function call to "send_email" with parameters: to (string), subject (string), body (string), cc (list of strings).',
    'Generate a JSON object representing a function call to "create_calendar_event" with parameters: title (string), date (string), time (string), duration_minutes (integer), attendees (list of strings).',
    'Generate a JSON object representing a function call to "get_weather" with parameters: city (string), country_code (string), units (string: "metric" or "imperial").',
    'Generate a JSON object representing a function call to "translate_text" with parameters: text (string), source_language (string), target_language (string).',
]

TOOL_INPUTS = [
    "The user wants to search for recent news about electric vehicles, showing 5 results with safe search on.",
    "Send an email to john@example.com about the project update, CC the manager at manager@example.com.",
    "Schedule a team standup meeting on 2024-05-10 at 9:00 AM for 30 minutes with Alice and Bob.",
    "Get the weather forecast for Paris, France in metric units.",
    "Translate 'Hello, how are you?' from English to Spanish.",
]

# ---------------------------------------------------------
# Task type registry
# ---------------------------------------------------------
TASK_TYPES = [
    {
        "name": "json_extraction",
        "instructions": EXTRACTION_INSTRUCTIONS,
        "inputs": EXTRACTION_INPUTS,
    },
    {
        "name": "schema_constrained_generation",
        "instructions": SCHEMA_INSTRUCTIONS,
        "inputs": SCHEMA_INPUTS,
    },
    {
        "name": "classification",
        "instructions": CLASSIFICATION_INSTRUCTIONS,
        "inputs": CLASSIFICATION_INPUTS,
    },
    {
        "name": "json_repair",
        "instructions": REPAIR_INSTRUCTIONS,
        "inputs": REPAIR_INPUTS,
    },
    {
        "name": "tool_call_generation",
        "instructions": TOOL_INSTRUCTIONS,
        "inputs": TOOL_INPUTS,
    },
]


# ---------------------------------------------------------
# API call helper
# ---------------------------------------------------------
def call_teacher(instruction: str, input_text: str) -> str | None:
    system_prompt = (
        "You are a helpful assistant that always responds with valid JSON only. "
        "Do not include any explanation, markdown, or extra text — output only the JSON object."
    )
    user_message = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{MAX_RETRIES}] Error: {e}")
            time.sleep(2)
    return None


def is_valid_json(text: str) -> bool:
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    return text


# ---------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------
def main():
    output_path = Path(OUTPUT_FILE)
    all_examples = []
    stats = {t["name"]: {"generated": 0, "valid": 0, "discarded": 0} for t in TASK_TYPES}

    print(f"Generating {N_PER_TASK_TYPE} examples per task type ({len(TASK_TYPES)} types = {N_PER_TASK_TYPE * len(TASK_TYPES)} total)")
    print(f"Teacher model: {MODEL}\n")

    for task in TASK_TYPES:
        task_name = task["name"]
        instructions = task["instructions"]
        inputs = task["inputs"]
        print(f"--- Task: {task_name} ---")

        count = 0
        attempt = 0
        max_attempts = N_PER_TASK_TYPE * 4  # allow retries

        while count < N_PER_TASK_TYPE and attempt < max_attempts:
            attempt += 1
            instruction = random.choice(instructions)
            input_text  = random.choice(inputs)

            stats[task_name]["generated"] += 1
            output = call_teacher(instruction, input_text)

            if output is None:
                stats[task_name]["discarded"] += 1
                continue

            cleaned = clean_json(output)
            if not is_valid_json(cleaned):
                print(f"  [INVALID JSON] Discarding: {cleaned[:80]}...")
                stats[task_name]["discarded"] += 1
                continue

            example = {
                "task_type":   task_name,
                "instruction": instruction,
                "input":       input_text,
                "output":      cleaned,
            }
            all_examples.append(example)
            stats[task_name]["valid"] += 1
            count += 1
            print(f"  [{count}/{N_PER_TASK_TYPE}] OK")
            time.sleep(SLEEP_BETWEEN)

        print(f"  Done: {stats[task_name]['valid']} valid, {stats[task_name]['discarded']} discarded\n")

    # Save as JSONL
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDataset saved to: {output_path}")
    print(f"Total examples: {len(all_examples)}")
    print("\nPer-task stats:")
    for name, s in stats.items():
        print(f"  {name}: {s['valid']} valid / {s['generated']} generated / {s['discarded']} discarded")


if __name__ == "__main__":
    main()
