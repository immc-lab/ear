import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from janus.models import MultiModalityCausalLM, VLChatProcessor
import PIL.Image
import numpy as np
import pandas as pd
from FInrTuner import FineTunedModel
import random
import argparse
import glob
import sys
import json


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(42)


def create_conversation(prompt):
    return [
        {
            "role": "<|User|>",
            "content": prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


def load_janus_model(model_path):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = model.to(torch.bfloat16).cuda().eval()
    return vl_chat_processor, model


@torch.inference_mode()
def generate(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        save_dir: str,
        temperature: float = 1,
        parallel_size: int = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        index: int = 0,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs(save_dir, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(save_dir, f"{index}.png")
        PIL.Image.fromarray(visual_img[i]).save(save_path)


def main(csv_path, output_dir, model_path, finetune_path=None):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_janus_model(model_path)

    finetuner = FineTunedModel(model.language_model.model, num_layers=5)
    finetuner.load_state_dict(torch.load(finetune_path))

    model.to(device)

    benchmark = pd.read_csv(csv_path)

    for idx, row in tqdm(benchmark.iterrows()):
        try:
            conversation = create_conversation(row["prompt"])

            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            final_prompt = sft_format + processor.image_start_tag

            with finetuner:

                generate(
                    model,
                    processor,
                    final_prompt,
                    save_dir=output_dir,
                    index=idx
                )
            print(f"Generated images for prompt: {row['prompt']} -> saved to {output_dir}")

        except Exception as e:
            print(f"Generation failed: {idx} prompt '{row['prompt']} '. Error:{str(e)}")


if __name__ == "__main__":
    config = {
        "csv_path": "../data/SIX-CD_Nudity.csv",
        "output_dir": "/path/to/save_path/ear_nudity/generated_imgs",
        "model_path": "/path/to/huggingface/Janus-Pro-7B",
        "finetune_path": "/path/to/save_path/nudity/ft_model_ear_nudity.pt"
    }

    main(**config)
