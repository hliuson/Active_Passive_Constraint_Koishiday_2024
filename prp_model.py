import os
import re
import numpy as np
import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from tqdm import tqdm
from score import score_relevance, score_APC

def load_generator(prp_scale):
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    prp_tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    prp_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
    
    prp_tokenizer.pad_token = prp_tokenizer.eos_token
    
    return prp_tokenizer, prp_model

def llama_msg_format(role, text):
    start_header_tok = '<|start_header_id|>'
    end_header_tok = '<|end_header_id|>'
    eot_tok = '<|eot_id|>'
    

    return f'{start_header_tok}{role}{end_header_tok}\n\n{text}{eot_tok}'

def retrieval_augmented_generate(character, statements, query, prp_model, prp_tokenizer, discriminator, rag_top_k):

    system_prompt = f"You are an AI agent role-playing as {character}, you should give a short response to the user's utterance as the character, not as an AI."
    sot_tok = '<|begin_of_text|>'
    start_header_tok = '<|start_header_id|>'
    end_header_tok = '<|end_header_id|>'
    eot_tok = '<|eot_id|>'
    
    
    scores = [score_relevance(character, statement, query, discriminator)[1].item() for statement in statements]
    retrieval_augmented_context = "\n".join([statements[idx] for idx in np.argsort(scores)[::-1][:rag_top_k]])
    
    system_prompt = retrieval_augmented_context+"\n"+system_prompt
    input_text = sot_tok + llama_msg_format("system", system_prompt) + llama_msg_format("user", query) + f'{start_header_tok}assistant{end_header_tok}\n\n'
    input_ids = prp_tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = prp_model.generate(**input_ids, max_new_tokens=1024, temperature=1.0, pad_token_id=prp_tokenizer.eos_token_id)
    
    response = prp_tokenizer.decode(outputs[0])
    response = response.replace(input_text, "")
    print(response)
    if not response.endswith(eot_tok):
        response = response + eot_tok
    
    return input_text, response

def generate_rag_dpo_dataset(character, prp_model, prp_tokenizer, relevance_discriminator, nli_discriminator, persona_statement_dataset, relevant_query_dataset, max_dpo_data, rag_top_k):

    if os.path.exists(f"statement/{character}.rag.dpo.json"):
        return json.load(open(f"statement/{character}.rag.dpo.json"))
    
    queries = [query for data in relevant_query_dataset for query in data["queries"]]
    queries = np.random.choice(queries, min(len(queries), max_dpo_data), replace=False)
    statements = [data["statement"] for data in persona_statement_dataset]

    dataset = []

    bar = tqdm(queries)

    with torch.no_grad():
        for query in bar:
            input_text, response_1 = retrieval_augmented_generate(character, statements, query, prp_model, prp_tokenizer, relevance_discriminator, rag_top_k)
            apc_1 = score_APC(character, statements, query, response_1, relevance_discriminator, nli_discriminator).item()
            input_text, response_2 = retrieval_augmented_generate(character, statements, query, prp_model, prp_tokenizer, relevance_discriminator, rag_top_k)
            apc_2 = score_APC(character, statements, query, response_2, relevance_discriminator, nli_discriminator).item()
            if np.abs(apc_1 - apc_2) > 0.2:
                chosen = [response_1, response_2][np.argmax([apc_1, apc_2])]
                rejected = [response_1, response_2][np.argmin([apc_1, apc_2])]
                dataset.append({"prompt": input_text, "chosen": chosen, "rejected": rejected})
            else:
                print(f"APC score difference is too small: {apc_1} vs {apc_2}")
                print(f"Query: {query}")
                print(f"Response 1: {response_1}")
                print(f"Response 2: {response_2}")

            bar.set_description(f"Generating APC-based RAG DPO Dataset... Number of Data: {len(dataset)}")
            
    json.dump(dataset, open(f"statement/{character}.rag.dpo.json", "w"))
            
    return dataset

def retrieval_augmented_generate_zh(character, statements, query, prp_model, prp_tokenizer, discriminator, rag_top_k):

    system_prompt = f"你是一个扮演“{character}”的AI, 你应当扮演“{character}”回应用户，而不是作为一个AI。"
    
    scores = [score_relevance(character, statement, query, discriminator)[1].item() for statement in statements]
    retrieval_augmented_context = "\n".join([statements[idx] for idx in np.argsort(scores)[::-1][:rag_top_k]])
    
    system_prompt = retrieval_augmented_context+"\n"+system_prompt
    input_text = f"<start_of_turn>model\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = prp_tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = prp_model.sample(**input_ids, max_new_tokens=256, temperature=1.0)
    response = prp_tokenizer.decode(outputs[0])
    if not response.endswith("<eos>"):
        response = response + "<eos>"
    response = re.findall("\n<start_of_turn>model\n(.*)?<eos>", response, re.DOTALL)[0] + "<eos>"
    
    return input_text, response

def train_prp(character, prp_model, prp_tokenizer, prp_scale, rag_dpo_dataset, lora_rank, prp_dpo_epoch):
    
    lora_config = LoraConfig(
        r=lora_rank,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    prp_model.add_adapter(lora_config)

    dpo_trainer = DPOTrainer(
        prp_model,
        ref_model=None,
        beta=0.1,
        train_dataset=Dataset.from_pandas(pd.DataFrame(rag_dpo_dataset)),
        tokenizer=prp_tokenizer,
        args=DPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=prp_dpo_epoch,
            learning_rate=2e-4,
            save_steps=10,
            fp16=True,
            logging_steps=1,
            output_dir=f"prp_models/llama-3-{prp_scale}-it-lora-{character}-rag-dpo",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
    )
    dpo_trainer.train()

    prp_tokenizer.save_pretrained(f"prp_models/llama-3-{prp_scale}-it-lora-{character}-rag-dpo")
    prp_model.save_pretrained(f"prp_models/llama-3-{prp_scale}-it-lora-{character}-rag-dpo")
    
    return prp_tokenizer, prp_model