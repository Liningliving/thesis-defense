# SPDX-License-Identifier: AGPL-3.0-only

from transformers import pipeline
from openie import StanfordOpenIE
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch

# Replace YOUR_TOKEN with your actual access token
login(token="")

device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import pipeline

# Initialize FLAN-T5-Base for text2text generation
summarizer = pipeline(
    "text2text-generation", 
    model="google/flan-t5-base", 
    tokenizer="google/flan-t5-base",
)

def summarize_object_descriptions(description_sentences,object_label):
    """
    summarize_descriptions using flan-t5-base.

    Parameters:
    - description_sentences: a stirng contian sentence to be summarized
    - object_label: The object label in the dataset must be contained in the summarized text

    Returns:
    - summary string
    """

    #The object label in the dataset must be contained in the summarized text
    # Build the prompt
    prompt = (
        "label: " + object_label + "\n"
        "Note: Do not use any pronouns (e.g., it, them) in your summary.\n"
        "summarize: " + description_sentences
    )

    output = summarizer(
    prompt, 
    max_length=60, 
    min_length=10
    )
    summary = output[0]["generated_text"]

    return summary                        

def summarize_relationship_descriptions(relationship_descriptions, objects_label):
    """
    summarize_descriptions using flan-t5-base.

    Parameters:
    - relationship_descriptions: a stirng contian sentence to be summarized
    - objects_label: str(obj0_label)+","+str(obj1_label)

    Returns:
    - summary string
    """
    
    #actual input
    prompt = (
        "labels: " + objects_label + "\n"
        "summarize: " + relationship_descriptions
    )

    output = summarizer(
    prompt, 
    max_length=60, 
    min_length=10
    )
    summary = output[0]["generated_text"]

    return summary