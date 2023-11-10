# from dataclasses import dataclass, field
# import logging

# from flask import Flask, request, jsonify
# import transformers

import os
import os.path

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

# from multi_token.training import (
#     ModelArguments,
# )
# from multi_token.inference import load_trained_lora_model
from multi_token.data_tools import encode_chat

# from typing import Type, List, Optional
# import logging

# from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from peft import PeftModel
# import torch

from multi_token.model_utils import fix_tokenizer
from multi_token.modalities.base_modality import Modality
from multi_token.language_models.mistral import MistralForCausalLM
from multi_token.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from multi_token.modalities import MODALITY_BUILDERS

# -- perplexity

from evaluate import load
# perplexity = load("perplexity", module_type="metric")
# results = perplexity.compute(predictions=predictions, model_id='gpt2')

# -- baseline

# import torch
# import torch.nn.functional as F

# from imagebind import data
# from imagebind.models.imagebind_model import ModalityType

MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda:0"

EMB_DIM = 4096
N_MODALITY_EMBS = 32

#PROMPT = "You are smart AI assistant. Please read the dialog and answer the question. Be short and precise!\n"
PROMPT = "This is a dialog with AI assistant.\n" # todo

DEBUG = False # True

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# import os.path
# import os

# from imagebind import data
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType

### from .utils import get_query_from_input, get_text_emb

# DEVICE = torch.device("cuda:0")
# DIALOGUE_DICT = {}

# bad_words_ids = tokenizer(["\nUser: ", "\n Bot:",], add_special_tokens=False).input_ids
#bad_words_ids = [
#    [29871, 13, 2659, 29901, 29871],
#    [29871, 13, 11273, 29901],
#]

#gen_params = {
#    "do_sample": False,
#    "max_new_tokens": 80,
#    "early_stopping": True,
#    "num_beams": 1,
#    "remove_invalid_values": True,
#    "eos_token_id": 29889,
#    "pad_token_id": 29889,
#    "forced_eos_token_id": 29889,
#    "use_cache": True,
#    "bad_words_ids": bad_words_ids,
#    "num_return_sequences": 1,
#}


#@torch.no_grad()
#def gen_answer(model, tokenizer, query, history=None):
#    query = torch.cat([history, query], dim=1)

#    out = model.generate(
#        inputs_embeds=query,
#        **gen_params,
#    )
#    out = out[:, 1:]
#    generated_texts = tokenizer.batch_decode(out)
#    return generated_texts[0]


#def imagebind_huge(pretrained=False):
#    model = imagebind_model.ImageBindModel(
#        vision_embed_dim=1280,
#        vision_num_blocks=32,
#        vision_num_heads=16,
#        text_embed_dim=1024,
#        text_num_blocks=24,
#        text_num_heads=16,
#        out_embed_dim=1024,
#        audio_drop_path=0.1,
#        imu_drop_path=0.7,
#    )

#    if pretrained:
#        model.load_state_dict(torch.load("/app/.checkpoints/imagebind_huge.pth"))

#    return model

# --- SETUP ---

# Function that returns model and tokenizer that will be used during the generation
def setup_model_and_tokenizer():

    print("\n=== SuperMachina v0.13 ===\n")

    print("Loading Mistral Instruct 7B v0.1...") # debug

    tokenizer = AutoTokenizer.from_pretrained("/app/mistral", padding_side="left", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("/app/mistral", torch_dtype=torch.float16).eval().to(DEVICE)

#    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", use_fast=False, cache_dir = "/app/models")
#    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, cache_dir = "/app/models").eval().to(DEVICE)

    print("Loading multimodal model...") # debug

    multimodel, multitokenizer = load_trained_lora_model(
        model_name_or_path = "/app/mistral", # MODEL, # serve_args.model_name_or_path,
        model_lora_path = "/app/lora", # "sshh12/Mistral-7B-LoRA-ImageBind-LLAVA", # serve_args.model_lora_path,
        load_bits = 16, # serve_args.load_bits,
        #cache_dir = "/app/models",
        device_map = DEVICE,
    )

    return [
        model,
        multimodel,
    ], tokenizer

#    return model, tokenizer



    # Instantiate model for image and audio embeddings
#    model_imagebind = imagebind_huge(pretrained=True).eval().to(DEVICE)
#    model_imagebind.query_dict = {}
        
#    EMB_DIM = 4096
#    N_MODALITY_EMBS = 32
#    ENC_DIM = model_imagebind.modality_heads[ModalityType.VISION][-1].out_features

#    projection = nn.Linear(ENC_DIM, N_MODALITY_EMBS * EMB_DIM).to(device=model.device, dtype=model.dtype).eval()
#    workdir = os.getcwd()

#    img_tokens_emb = torch.load(
#        f"{workdir}/team_code/ckpts/IMG_EMB_LLaMa-7b-EN-Linear-ImageBind",
#        map_location=model.device,
#    )
#    audio_tokens_emb = torch.load(
#        f"{workdir}/team_code/ckpts/AUDIO_EMB_LLaMa-7b-EN-Linear-ImageBind",
#        map_location=model.device,
#    )
#    projection = torch.load(
#        f"{workdir}/team_code/ckpts/projection_LLaMa-7b-EN-Linear-ImageBind",
#        map_location=model.device,
#    )

#    return [
#        model,
#        model_imagebind,
#        img_tokens_emb,
#        audio_tokens_emb,
#        projection,
#    ], tokenizer

# --- GENERATE ---

# Function that generates the responses for dialodues queries w.r.t. history.
def generate_text(model, tokenizer, cur_query_list, history_tensor=None):

    num = 0 # number of current iteration in history

    for part in cur_query_list:
        if part["type"] == "text":
            prompt = part["content"]
            #history_tensor[0][num]["prompt"] = prompt 

    # -- handle history

    # If the current history is empty - it is assigned to the system prompt
    if history_tensor is None:
        # PROMPT = "This is a dialog with AI assistant.\n"
        # prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        ### prompt_embeddings = model[0](prompt_ids).logits

        ### print("\n=== prompt_embeddings ===\n", prompt_embeddings)

        ### prompt_embeddings = model[0].embed_tokens(prompt_ids)
        # history_tensor = prompt_embeddings
# debug        history_tensor = get_text_emb(model[0], tokenizer, PROMPT)
        # debug
        history_tensor = ([
            {
                "id": "",
                "session": "",
                "prompt": prompt,
                "response": "",
                "embd": "", # prompt_embeddings
                "history": PROMPT  + "\nUser: " + prompt,
            }
        ], "")

    else:
        # print("\n === GET TEXT HISTORY ===\n", history_tensor) # debug
        num = len(history_tensor[0])
        # embd = torch.concat(
        #     [
        #         history_tensor[0][num-1]["embd"],
        #         get_text_emb(model[0], tokenizer, history_tensor[1])
        #     ], dim=1)

        # -- update history

        history_tensor[0].append(
            {
                "id": "",
                "session": "",
                "prompt": prompt,
                "response": "",
                "embd": "", # embd
                "history": history_tensor[0][num-1]["history"] + "\nUser: " + prompt,
            })
        
    print("\n === HISTORY ===\n", history_tensor) # debug    
        
   
        
    # -- handle query    

    #    json={
    #       "messages": [{"role": "user", "content": "<imagebind> What is the animal in this sound?"}],
    #       "imagebinds": ["https://github.com/sshh12/multi_token/raw/main/.demo/imagebind-dog-audio.wav"],
    #    },

    # media = ""
    media = "<imagebind> "
    query = {
        "messages": [],
        "imagebinds": [ "/app/transparent.png" ],
    } 

    if len(cur_query_list) > 1:
        media = "<imagebind> "
        query = {
            "messages": [],
            "imagebinds": [],
        } 

    for el in cur_query_list:
        if el["type"] == "text":
            query["messages"].append({
                "role": "user", 
                "content": media + el["content"]
            })
        else:    
            query["imagebinds"].append(
                el["content"]
            )

    print("\n=== query ===\n", query)

    encoded_dict = encode_chat(query, tokenizer, model[1].modalities)

    # print("\n=== encoded_dict ===\n", encoded_dict)

    # -- generate

    with torch.inference_mode():
        output_ids = model[1].generate(

            input_ids = encoded_dict["input_ids"].unsqueeze(0).to(model[1].device),
            max_new_tokens = 400, # serve_args.max_new_tokens,
            use_cache = True,
            do_sample = True,
            temperature = 0.1, # serve_args.temperature,

            modality_inputs = {
                m.name: [encoded_dict[m.name]] for m in model[1].modalities
            },

            pad_token_id=tokenizer.eos_token_id, # debug
        )

    response = tokenizer.decode(
        output_ids[0, encoded_dict["input_ids"].shape[0] :],
        skip_special_tokens=True,
    ).strip()

    print("\n=== response ===\n", response)

    # -- update history and return results

    # FIXME: Update history ?!
    history_tensor[0][num]["response"] = response
    history_tensor[0][num]["history"] = history_tensor[0][num]["history"] + "\nBot: " + response
    # prompt = get_query_from_input(model[1], tokenizer, cur_query_list).to(DEVICE)
    # history_tensor[0][num]["embd"] = torch.concat([history_tensor[0][num]["embd"], prompt], dim=1)

    return response, history_tensor[0]

    # return outputs, []

#    if history_tensor is not None:
#        history_tensor = torch.concat(
#            [history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])],
#            dim=1,
#        )
#    else:
        # If the current history is empty
        # it is assigned to the system prompt
#        PROMPT = "This is a dialog with AI assistant.\n"
#        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
#        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
#        history_tensor = prompt_embeddings

#    prompt = get_query_from_input(model, tokenizer, cur_query_list).to(DEVICE)
#    response = gen_answer(model[0], tokenizer, prompt, history=history_tensor)

#    history_tensor = torch.concat([history_tensor, prompt], dim=1)

#    return response, history_tensor

# --- PPL ---

# FIXME: Histroy Tensor !!!

def get_ppl(model, tokenizer, cur_query_tuple, history_tensor=None):

    # print("\n === PPL HISTORY ===\n", history_tensor) # debug

#    if history_tensor is None:
#        history_tensor = ([
#            {
#                "id": "",
#                "session": "",
#                "prompt": "",
#                "response": "",
#                "embd": "", # prompt_embeddings
#            }
#        ], "")

        # If the current history is empty - it is assigned to the system prompt
        # prompt = "This is a dialog with AI assistant.\n" # todo
        
        # prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        # prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        # history_tensor = prompt_embeddings

#    else:
        # num = len(history_tensor[0])
        #history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
        # history_tensor = torch.concat([history_tensor[0][num-1]["embd"], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)

#        history_tensor = ([
#            {
#                "id": "",
#                "session": "",
#                "prompt": "",
#                "response": "",
#                "embd": "", # prompt_embeddings
#            }
#        ], "")


#ValueError: Module inputs don't match the expected format.
#Expected format: {'predictions': Value(dtype='string', id=None)},
#Input predictions: [[{'type': 'text', 'content': 'What common between those picture and sounds?'}, {'type': 'image', 'content': '00001.png'}, {'type': 'audio', 'content': '00000.wav'}], 'The Pacific Ocean is the largest and deepest of the world ocean basins.']

#    question = ""
#    for part in cur_query_tuple[0]:
#        if part["type"] == "text":
#            question = part["content"]

    data = [ 
        history_tensor[1], # model answer
        cur_query_tuple[1], # suggested answer
    ]

    print("\n=== DATA ===\n", data)

    # perplexity = load("perplexity", module_type="metric", cache_dir = "/app/models")
    perplexity = load("perplexity", module_type="metric")

    result = perplexity.compute(
        predictions = data, 
        model_id = "/app/mistral", #  MODEL,
        # cache_dir = "/app/models",
    )

    print("\n=== RESULT ===\n", result)

    return result['perplexities'] # result.mean_perplexity




#    current_query = get_query_from_input(model[0], tokenizer, cur_query_tuple[0])
#    current_answer = get_text_emb(model[0], tokenizer, cur_query_tuple[1])

    # Input dialogue query with history
#    dialogue_emb = torch.concat([history_tensor, current_query], dim=1).to(DEVICE)
#    inputs_embeds=torch.concat([dialogue_emb, current_answer], dim=1)
    
#    loss = nn.CrossEntropyLoss()
#    with torch.no_grad():
#        out_logits = model[0](inputs_embeds=inputs_embeds).logits

#    shift_logits = out_logits[..., : -1, :].contiguous()
#    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
#    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
#    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
#    shift_labels = labels[..., 1:].contiguous()
    
#    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
#    ppl = torch.exp2(neg_log_likelihood)
    
#    return ppl.item(), dialogue_emb



# -- LOAD

def load_trained_lora_model(
    model_name_or_path: str,
    model_lora_path: str,
#    model_cls: Optional[Type] = None,
#    modalities: Optional[List[Modality]] = None,
    load_bits: int = 16,
    device_map: str = "auto",
#    cache_dir: str = "",
):
    load_kwargs = {"device_map": device_map}

#    if load_bits == 8:
#        load_kwargs["load_in_8bit"] = True
#    elif load_bits == 4:
#        load_kwargs["load_in_4bit"] = True
#        load_kwargs["quantization_config"] = BitsAndBytesConfig(
#            load_in_4bit=True,
#            bnb_4bit_compute_dtype=torch.float16,
#            bnb_4bit_use_double_quant=True,
#            bnb_4bit_quant_type="nf4",
#        )
#    elif load_bits == 16:
    load_kwargs["torch_dtype"] = torch.float16
#    else:
#        raise ValueError(f"Invalid load_bits: {load_bits}")

    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, cache_dir = cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    fix_tokenizer(tokenizer) # FIXME

    cfg = AutoConfig.from_pretrained(model_lora_path)
#    if model_cls is None:
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[cfg.model_cls]
#    if modalities is None:
    modalities = MODALITY_BUILDERS[cfg.modality_builder]()

    print(f"Loading base model from {model_name_or_path} as {load_bits} bits")
    model = model_cls.from_pretrained(
        #model_name_or_path, low_cpu_mem_usage=True, config=cfg, **load_kwargs, cache_dir = cache_dir
        model_name_or_path, low_cpu_mem_usage=True, config=cfg, **load_kwargs
    )
    model.modalities = modalities

    # FIXME: Dowmload local !!!
    print(f"Loading projector weights for {[m.name for m in modalities]}")
    if os.path.exists(os.path.join(model_lora_path, "non_lora_trainables.bin")):
        non_lora_trainables = torch.load(
            os.path.join(model_lora_path, "non_lora_trainables.bin"), map_location="cpu"
        )
    else:
        local_fn = hf_hub_download(
            repo_id=model_lora_path,
            filename="non_lora_trainables.bin",
            repo_type="model",
        )
        non_lora_trainables = torch.load(local_fn, map_location="cpu")
    model.get_model().initialize_pretrained_modules(modalities, non_lora_trainables)

    print(f"Loading and merging LoRA weights from {model_lora_path}")
    model = PeftModel.from_pretrained(model, model_lora_path)
    if load_bits == 16:
        # TODO: Figure out why this fails for other bit sizes
        model = model.merge_and_unload()
    model.eval()

    return model, tokenizer
