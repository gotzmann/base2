from dataclasses import dataclass, field
import logging

# from flask import Flask, request, jsonify
import transformers
import torch

from multi_token.training import (
    ModelArguments,
)
from multi_token.inference import load_trained_lora_model
from multi_token.data_tools import encode_chat


from typing import Type, List, Optional
# import logging

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from peft import PeftModel
import torch
import os

from multi_token.model_utils import fix_tokenizer
from multi_token.modalities.base_modality import Modality
from multi_token.language_models.mistral import MistralForCausalLM
from multi_token.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from multi_token.modalities import MODALITY_BUILDERS

# -- baseline

# import torch
import torch.nn.functional as F

# from imagebind import data
# from imagebind.models.imagebind_model import ModalityType

DEVICE = "cuda:0"
EMB_DIM = 4096
N_MODALITY_EMBS = 32

import torch.nn as nn
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os.path
# import os

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# from .utils import get_query_from_input, get_text_emb

# DEVICE = torch.device("cuda:0")
DIALOGUE_DICT = {}

# bad_words_ids = tokenizer(["\nUser: ", "\n Bot:",], add_special_tokens=False).input_ids
bad_words_ids = [
    [29871, 13, 2659, 29901, 29871],
    [29871, 13, 11273, 29901],
]

gen_params = {
    "do_sample": False,
    "max_new_tokens": 80,
    "early_stopping": True,
    "num_beams": 1,
    "remove_invalid_values": True,
    "eos_token_id": 29889,
    "pad_token_id": 29889,
    "forced_eos_token_id": 29889,
    "use_cache": True,
    "bad_words_ids": bad_words_ids,
    "num_return_sequences": 1,
}


@torch.no_grad()
def gen_answer(model, tokenizer, query, history=None):
    query = torch.cat([history, query], dim=1)

    out = model.generate(
        inputs_embeds=query,
        **gen_params,
    )
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)
    return generated_texts[0]


def imagebind_huge(pretrained=False):
    model = imagebind_model.ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        model.load_state_dict(torch.load("/app/.checkpoints/imagebind_huge.pth"))

    return model


# Function that returns model and tokenizer that will be used during the generation
def setup_model_and_tokenizer():

    model, tokenizer = load_trained_lora_model(
        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1", # serve_args.model_name_or_path,
        model_lora_path = "sshh12/Mistral-7B-LoRA-ImageBind-LLAVA", # serve_args.model_lora_path,
        load_bits = 16, # serve_args.load_bits,
    )

    return model, tokenizer

#    tokenizer = AutoTokenizer.from_pretrained("/app/Llama-2-7B-fp16", padding_side="left", use_fast=False)
#    model = AutoModelForCausalLM.from_pretrained("/app/Llama-2-7B-fp16", torch_dtype=torch.float16).eval().to(DEVICE)

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

    #    json={
    #       "messages": [{"role": "user", "content": "<imagebind> What is the animal in this sound?"}],
    #       "imagebinds": ["https://github.com/sshh12/multi_token/raw/main/.demo/imagebind-dog-audio.wav"],
    #    },

    query = {
         "messages": [],
         "imagebinds": [],
    }

    media = ""
    if len(cur_query_list) > 1:
        media = "<imagebind> "  

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


    encoded_dict = encode_chat(query, tokenizer, model.modalities)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids = encoded_dict["input_ids"].unsqueeze(0).to(model.device),
            max_new_tokens = 400, # serve_args.max_new_tokens,
            use_cache = True,
            do_sample = True,
            temperature = 0.1, # serve_args.temperature,
            modality_inputs = {
                m.name: [encoded_dict[m.name]] for m in model.modalities
            },
        )

    outputs = tokenizer.decode(
        output_ids[0, encoded_dict["input_ids"].shape[0] :],
        skip_special_tokens=True,
    ).strip()

    return outputs, []

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

def get_ppl(model, tokenizer, cur_query_tuple, history_tensor=None):

#    if history_tensor is not None:
#        history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
#    else:
        # If the current history is empty
        # it is assigned to the system prompt
#        PROMPT = "This is a dialog with AI assistant.\n"
#        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
#        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
#        history_tensor = prompt_embeddings

    # print("\n === PPL HISTORY ===\n", history_tensor) # debug

    if history_tensor is None:

        # If the current history is empty - it is assigned to the system prompt
        prompt = "This is a dialog with AI assistant.\n" # todo
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    else:
        num = len(history_tensor[0])
        #history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
        history_tensor = torch.concat([history_tensor[0][num-1]["embd"], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)

    current_query = get_query_from_input(model, tokenizer, cur_query_tuple[0])
    current_answer = get_text_emb(model[0], tokenizer, cur_query_tuple[1])

    # Input dialogue query with history
    dialogue_emb = torch.concat([history_tensor, current_query], dim=1).to(DEVICE)
    inputs_embeds=torch.concat([dialogue_emb, current_answer], dim=1)
    
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        out_logits = model[0](inputs_embeds=inputs_embeds).logits

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb


# === UTILS ===


# utils function that parses the format of the input query to a single sequence
def get_query_from_input(model, tokenizer, input_list):
    base_model = model[0]
    model_imagebind = model[1]
    img_tokens_emb = model[2]
    audio_tokens_emb = model[3]
    projection = model[4]

    all_emb = []

    ai_ids = tokenizer.encode("\n Bot: ", add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ai_embeddings = base_model.model.embed_tokens(ai_ids)

    prompt = "\nUser: "
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    prompt_embeddings = base_model.model.embed_tokens(prompt_ids)
    all_emb.append(prompt_embeddings)

    for el in input_list:
        if el["type"] == "text":
            query = el["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(DEVICE)
            query_embeddings = base_model.model.embed_tokens(query_ids)
            all_emb.append(query_embeddings)
        elif el["type"] == "image":
            modality_start_emb, modality_end_emb = img_tokens_emb
            filepath = f"{el['content']}"
            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_image(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs
            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM), 
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )
        else:
            modality_start_emb, modality_end_emb = audio_tokens_emb
            filepath = f"{el['content']}"
            if filepath in model_imagebind.query_dict:
                projected_modality_embs = model_imagebind.query_dict[filepath]
            else:
                modality_embedding = encode_audio(model_imagebind, filepath).to(device=base_model.device, dtype=base_model.dtype)
                projected_modality_embs = projection(modality_embedding).to(device=base_model.device, dtype=base_model.dtype)
                model_imagebind.query_dict[filepath] = projected_modality_embs
            all_emb.extend(
                [
                    modality_start_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                    projected_modality_embs.reshape(1, N_MODALITY_EMBS, EMB_DIM), 
                    modality_end_emb[None, None].to(device=base_model.device, dtype=base_model.dtype),
                ]
            )

        all_emb.append(ai_embeddings)

        embeddings = torch.cat(
            all_emb,
            dim=1,
        )
    return embeddings


def get_text_emb(model, tokenizer, text):
    if text is None or len(text) == 0:
        text = "I don't know.\n"
    text_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    text_embeddings = model.model.embed_tokens(text_ids)
    return text_embeddings


@torch.no_grad()
def encode_audio(model_imagebind, audio_paths, normalize=True):
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths=audio_paths, device=DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.AUDIO].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings


@torch.no_grad()
def encode_image(model_imagebind, image_paths, normalize=True):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, DEVICE),
    }
    universal_embeddings = model_imagebind(inputs)[ModalityType.VISION].to(DEVICE)
    if normalize:
        universal_embeddings = F.normalize(universal_embeddings, dim=-1)
    return universal_embeddings
