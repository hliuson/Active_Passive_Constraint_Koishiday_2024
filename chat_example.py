import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prp_model import retrieval_augmented_generate
from classifier import get_relevance_discriminator
import gradio as gr

character = "Alice"

statements = [data["statement"] for data in json.load(open(f"statement/{character}.json"))]

model_id = f"prp_models/llama-3-7b-it-lora-{character}-rag-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

prp_tokenizer = AutoTokenizer.from_pretrained(model_id)
prp_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

relevance_discriminator = get_relevance_discriminator(character=None, statement_query_relevance_dataset=None, relevance_finetune_epoch=None, use_pretrained_discriminator=True)

print(f"You are chatting with {character}!")

def chat_interface(user_input):
    with torch.no_grad():
        _, response = retrieval_augmented_generate(character, statements, user_input, prp_model, prp_tokenizer, relevance_discriminator, rag_top_k=5)
        response = character + ": " + response.replace("<eos>", "")
        return response

iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text", title=model_id)
iface.launch(share=True)
