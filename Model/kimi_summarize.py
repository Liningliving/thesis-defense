# SPDX-License-Identifier: AGPL-3.0-only

import time
from openai import OpenAI
import json
import re
import ast
from openai import RateLimitError

api_key = ""  # Replace with your actual API Key
base_url = "https://api.moonshot.cn/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

functions = [{
    "name": "format_relations",
    "description": "Return a JSON list of [subject, relation, object] triplets",
    "parameters": {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3
                }
            }
        },
        "required": ["relations"]
    }
}]

def extract_relationships_via_function(content: str, max_retries=5):
    '''system_msg = (
        "You are a Spatial‐Relation Extractor. "
        "You MUST NOT output any plain text. "
        "You must call the function `format_relations` "
        "and pass exactly one argument `relations` which is an array of [subject, relation, object] triplets."
    )'''

    system_msg = (
    "You are a Spatial-Relation Extractor. "
    "First remove duplicate or redundant sentences and summarize the context. "
    "Then extract *only* spatial relations and output them by calling the function "
    "`format_relations`—no other text, no explanations, no labels. "
    "The call must look exactly like: "
    "format_relations({\"relations\": [[\"subject\",\"relation\",\"object\"], …]})\n\n"
    "Example:\n"
    "Input: \"The cup is on the table. The plate sits below the lamp.\"\n"
    "Output: format_relations([[\"cup\",\"on\",\"table\"],[\"plate\",\"below\",\"lamp\"]])"
    )

    for attempt in range(1, max_retries+1):
        try:
            res = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system",  "content": system_msg},
                    {"role": "user",    "content": content},
                ],
                temperature=0.1,
                # top_k=1,         # force greedy
                # top_p=1.0,       # no nucleus sampling
                seed=1314520,       # <- IMPORTANT
                functions=functions,
                # function_call="auto",
                function_call={"name": "extract_relationships_via_function"},  
                max_tokens= 512,
            )
            msg = res.choices[0].message
            # print(msg)
            # Extract the list of lists from the string
            # msg = ast.literal_eval(msg.content.split('(', 1)[1].rsplit(')', 1)[0])
            # relatoins = json.dumps(msg, indent=2)
            # return msg
        
            # 1) Preferred: proper function_call
            if msg.function_call:
                return json.loads(msg.function_call.arguments)["relations"]

            raw = msg.content

            # 2) format_relations([...]) wrapper
            m_fn = re.search(r"format_relations\(\s*(\[\s*\[.*?\]\s*\])\s*\)", raw, re.DOTALL)
            if m_fn:
                return json.loads(m_fn.group(1))
            
            m_empty = re.search(r"format_relations\(\s*\[\s*\]\s*\)", raw)
            if m_empty:
                return []

            # 3) plain JSON array
            m_json = re.search(r"(\[\s*\[.*?\]\s*\])", raw, re.DOTALL)
            if m_json:
                return json.loads(m_json.group(1))

            # Nothing matched—dump and retry
            print(f"[Attempt {attempt}] No usable output. Full response:\n{res}")
            raise RuntimeError("No relations output detected")

        except Exception as e:
            # print(f"[Attempt {attempt}] Error: {e}; retrying…")
            # time.sleep(1)
            wait = getattr(e, "retry_after", None) or 20  # seconds
            # print(f"Rate-limited—sleeping {wait}s…")
            time.sleep(wait)
    raise RuntimeError("Failed to extract relationships after retries")

'''
def extract_relationship(content: str, max_retries=5, temperature=0.0,
    do_sample=False,
    top_p=1.0,
    top_k=0,
    n=1,):
    delay = 1  # start with 1 second
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {
                        "role": "system",
                        "content": "You are good at retrieving relationship information from sentences."
                    },
                    {
                        "role": "user",
                        "content": 'First summarize the sentences, then extract all clear and grammatically correct spatial relations from the sentences I provide in the format: ["subject", "relation", "object"],...,["subject", "relation", "object"]. \
                            Ignore unclear, contradictory, or grammatically incorrect statements. Output only these Spatial relations.\
                            The sentences are: ' + content
                    },
                ],
                temperature=temperature,
                )
            answer = completion.choices[0].message
            return answer.content
        
        except :
            pass
'''
