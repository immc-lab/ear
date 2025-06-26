import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from FInrTuner import FineTunedModel
import os
import random
import numpy as np


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(42)

model_path = "/path/to/huggingface/Janus-Pro-7B"
train_data_path = "../data/train_nudity.json"
path_base = "/path/to/save_path"
save_path = f"{path_base}/nudity/ft_model_ear_nudity.pt"

lr = 1e-4
iterations = 50
negative_guidance = 1.0
accumulation_steps = 100

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
model = vl_gpt.language_model.model


def get_embdding(prompt, max_length=None):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    if max_length is not None:
        input_ids = input_ids[:max_length]
        input_ids += [vl_chat_processor.pad_id] * (max_length - len(input_ids))
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = model.get_input_embeddings()(tokens)
    return inputs_embeds


finetuner = FineTunedModel(model, num_layers=5)

print("started training", save_path)

optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
criteria = torch.nn.MSELoss()
pbar = tqdm(range(iterations))

with open(train_data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

unsafe_prompts = [item.get('original_prompt') for item in data if 'original_prompt' in item]
safe_prompts = [item.get('modified_prompt') for item in data if 'modified_prompt' in item]

os.makedirs(os.path.dirname(save_path), exist_ok=True)

for _ in pbar:
    print(f"No.{_ + 1} batch start：")
    accumulated_loss = 0
    positive_text_embeddings = get_embdding(unsafe_prompts[_])
    neutral_text_embeddings = get_embdding(safe_prompts[_],
                                           max_length=len(vl_chat_processor.tokenizer.encode(unsafe_prompts[_])))
    target_text_embeddings = get_embdding(safe_prompts[_],
                                          max_length=len(vl_chat_processor.tokenizer.encode(unsafe_prompts[_])))

    tqdms = tqdm(range(576))
    for i in tqdms:
        positive_outputs = model(inputs_embeds=positive_text_embeddings)
        neutral_outputs = model(inputs_embeds=neutral_text_embeddings)
        target_outputs = model(inputs_embeds=target_text_embeddings)

        with finetuner:
            negative_outputs = model(inputs_embeds=positive_text_embeddings)

        positive_hidden_states = positive_outputs.last_hidden_state.detach()
        neutral_hidden_states = neutral_outputs.last_hidden_state.detach()
        target_hidden_states = target_outputs.last_hidden_state.detach()
        negative_hidden_states = negative_outputs.last_hidden_state

        loss = criteria(negative_hidden_states,
                        target_hidden_states - (negative_guidance * (positive_hidden_states - neutral_hidden_states)))
        accumulated_loss += loss
        tqdms.set_postfix({"loss": accumulated_loss.item()})

        num_steps = i + 1
        if num_steps % accumulation_steps == 0:
            if accumulated_loss > 0.05:
                optimizer.zero_grad()
                accumulated_loss.backward()
                optimizer.step()
                accumulated_loss = 0.0
            else:
                print("The loss is less than 0.05, the gradient update is abandoned")
                optimizer.zero_grad()
                accumulated_loss = 0.0

        if num_steps in [286, 576]:
            intermediate_save_path = f"{path_base}/ft_model_ear_nudity_d{_ + 1}_i{num_steps}.pt"
            torch.save(finetuner.state_dict(), intermediate_save_path)
            print(f"model saved to: {intermediate_save_path}")

        positive_logits = vl_gpt.gen_head(positive_hidden_states[:, -1, :])
        neutral_logits = vl_gpt.gen_head(neutral_hidden_states[:, -1, :])
        target_logits = vl_gpt.gen_head(target_hidden_states[:, -1, :])

        positive_logit_cond = positive_logits[0::2, :]
        neutral_logit_cond = neutral_logits[0::2, :]
        target_logit_cond = target_logits[0::2, :]

        positive_logit_uncond = positive_logits[1::2, :]
        neutral_logit_uncond = neutral_logits[1::2, :]
        target_logit_uncond = target_logits[1::2, :]

        positive_logits = positive_logit_uncond + 5 * (positive_logit_cond - positive_logit_uncond)
        neutral_logits = neutral_logit_uncond + 5 * (neutral_logit_cond - neutral_logit_uncond)
        target_logits = target_logit_uncond + 5 * (target_logit_cond - target_logit_uncond)

        positive_probs = torch.softmax(positive_logits, dim=-1)
        neutral_probs = torch.softmax(neutral_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

        positive_next_token = torch.multinomial(positive_probs, num_samples=1)
        neutral_next_token = torch.multinomial(neutral_probs, num_samples=1)
        target_next_token = torch.multinomial(target_probs, num_samples=1)

        positive_next_token = torch.cat([positive_next_token.unsqueeze(dim=1), positive_next_token.unsqueeze(dim=1)],
                                        dim=1).view(-1)
        neutral_next_token = torch.cat([neutral_next_token.unsqueeze(dim=1), neutral_next_token.unsqueeze(dim=1)],
                                       dim=1).view(-1)
        target_next_token = torch.cat([target_next_token.unsqueeze(dim=1), target_next_token.unsqueeze(dim=1)],
                                      dim=1).view(-1)

        positive_img_embeds = vl_gpt.prepare_gen_img_embeds(positive_next_token)
        neutral_img_embeds = vl_gpt.prepare_gen_img_embeds(neutral_next_token)
        target_img_embeds = vl_gpt.prepare_gen_img_embeds(target_next_token)

        positive_text_embeddings = positive_img_embeds.unsqueeze(dim=1)
        neutral_text_embeddings = neutral_img_embeds.unsqueeze(dim=1)
        target_text_embeddings = target_img_embeds.unsqueeze(dim=1)

torch.save(finetuner.state_dict(), save_path)
print("model saved to: ", save_path)

torch.cuda.empty_cache()
