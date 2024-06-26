import argparse

parser = argparse.ArgumentParser(description='Parser for APC-based DPO.')

parser.add_argument('--api_key', type=str)
parser.add_argument('--hf_token', type=str)
parser.add_argument('--character', type=str)
parser.add_argument('--model_engine', type=str)
parser.add_argument('--use_pretrained_discriminator', action='store_true')
parser.add_argument('--relevance_finetune_epoch', type=int)
parser.add_argument('--rag_top_k', type=int)
parser.add_argument('--nli_finetune_epoch', type=int)
parser.add_argument('--max_dpo_data', type=int)
parser.add_argument('--lora_rank', type=int)
parser.add_argument('--prp_dpo_epoch', type=int)
parser.add_argument('--prp_scale', type=str)
parser.add_argument('--device', type=str)

args = parser.parse_args()

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["WANDB_DISABLED"] = "true"

from openai import AzureOpenAI
import json
import re
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from datasets import Dataset

import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model, get_peft_model_state_dict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from classifier import Classifier, get_relevance_discriminator, get_nli_discriminator
from prompt import (convert_to_statement, build_relevant_query_dataset, build_statement_query_relevance_dataset, 
                   build_statement_to_response_nli_dataset, discriminate_statement_to_response_nli_dataset)
from score import score_APC
from prp_model import load_generator, generate_rag_dpo_dataset, train_prp

if not os.path.exists("statement"):
    os.makedirs("statement")


#check gpu

character = args.character
model_engine = args.model_engine
use_pretrained_discriminator = args.use_pretrained_discriminator
relevance_finetune_epoch = args.relevance_finetune_epoch
rag_top_k = args.rag_top_k
nli_finetune_epoch = args.nli_finetune_epoch
max_dpo_data = args.max_dpo_data
lora_rank = args.lora_rank
prp_dpo_epoch = args.prp_dpo_epoch
prp_scale = args.prp_scale

client = AzureOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                     azure_endpoint=os.environ.get("OPENAI_API_BASE"),
                     api_version=os.environ.get("OPENAI_API_VERSION"))

# Stage 1: Dataset Synthesis

persona_statement_dataset = convert_to_statement(character, model_engine, client)
relevant_query_dataset = build_relevant_query_dataset(character, persona_statement_dataset, model_engine, client)
if not use_pretrained_discriminator:
    statement_query_relevance_dataset = build_statement_query_relevance_dataset(character, relevant_query_dataset, model_engine, client)
    statement_to_response_nli_dataset = build_statement_to_response_nli_dataset(character, relevant_query_dataset, model_engine, client)
    statement_to_response_nli_v2_dataset = discriminate_statement_to_response_nli_dataset(character, statement_to_response_nli_dataset, model_engine, client)
else:
    statement_query_relevance_dataset = None
    statement_to_response_nli_dataset = None
    statement_to_response_nli_v2_dataset = None

# Stage 2: Discriminator Fine-tuning

relevance_discriminator = get_relevance_discriminator(character, statement_query_relevance_dataset, relevance_finetune_epoch, use_pretrained_discriminator)
nli_discriminator = get_nli_discriminator(character, statement_to_response_nli_v2_dataset, nli_finetune_epoch, use_pretrained_discriminator)

# Stage 3: APC-based DPO

prp_tokenizer, prp_model = load_generator(prp_scale)
rag_dpo_dataset = generate_rag_dpo_dataset(character, prp_model, prp_tokenizer, relevance_discriminator, nli_discriminator, persona_statement_dataset, relevant_query_dataset, max_dpo_data, rag_top_k)
prp_tokenizer, prp_model = train_prp(character, prp_model, prp_tokenizer, prp_scale, rag_dpo_dataset, lora_rank, prp_dpo_epoch)
