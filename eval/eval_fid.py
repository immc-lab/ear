import argparse
from argparse import ArgumentParser
import gc
from pathlib import Path
import warnings
import torch
from PIL import Image
import PIL.Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from accelerate import PartialState, Accelerator
from tqdm import tqdm
from janus.models import MultiModalityCausalLM, VLChatProcessor
from peft import PeftModel
from FInrTuner import FineTunedModel
from pydantic import BaseModel
import json
import pandas as pd
from prettytable import PrettyTable
from cleanfid import fid
import os
from typing import Iterator, List
import random


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(42)


class GenerationConfig(BaseModel):
    prompts: list[str] = []
    negative_prompt: str = (
        "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    )
    unconditional_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 2024
    generate_num: int = 1
    save_path: str = None

    def dict(self):
        results = {}
        for attr in vars(self):
            if not attr.startswith("_"):
                results[attr] = getattr(self, attr)
        return results

    @staticmethod
    def fix_format(cfg):
        for k, v in cfg.items():
            if isinstance(v, list):
                cfg[k] = v[0]
            elif isinstance(v, torch.Tensor):
                cfg[k] = v.item()


class Coco30kGenerationDataset(IterableDataset):
    def __init__(
            self,
            save_folder: str = "/path/to/coco30k_fid/generated_imgs/",
            base_cfg: GenerationConfig = GenerationConfig(),
            data_path: str = "../data/coco_30k.csv",
            batch_size: int = 4,
            **kwargs,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path)
        self.data = []
        self.batch_size = batch_size

        # Group data into batches
        batch = []
        for idx, row in df.iterrows():
            cfg = base_cfg.copy()
            cfg.prompts = [row["prompt"]]
            cfg.negative_prompt = ""
            cfg.width = row["width"] - row["width"] % 8
            cfg.height = row["height"] - row["height"] % 8
            cfg.seed = row["evaluation_seed"]
            cfg.generate_num = 1
            cfg.save_path = os.path.join(
                save_folder,
                "coco30k",
                "COCO_val2014_" + "%012d" % row["image_id"] + ".jpg",
            )
            batch.append(cfg.dict())

            if len(batch) >= batch_size:
                self.data.append(batch)
                batch = []

        if batch:
            self.data.append(batch)

    def __iter__(self) -> Iterator[List[dict]]:
        return iter(self.data)


class CocoEvaluator:

    def __init__(
            self,
            save_folder: str = "/path/to/coco30k_fid/generated_imgs/",
            output_path: str = "/path/to/coco30k_fid/results/",
            data_path: str = "/path/to/COCO/val2014",
    ):
        self.save_folder = save_folder
        self.output_path = output_path
        self.data_path = data_path

        if not os.path.exists(self.save_folder):
            raise FileNotFoundError(f"Image path {self.save_folder} not found.")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def evaluation(self):
        print("Evaluating on COCO-30k Caption dataset...")
        fid_value = fid.compute_fid(
            os.path.join(self.save_folder, "coco30k"), self.data_path, num_workers=0
        )

        pt = PrettyTable()
        pt.field_names = ["Metric", "Value"]
        pt.add_row(["FID", fid_value])
        print(pt)
        with open(os.path.join(self.output_path, "coco-fid.json"), "w") as f:
            json.dump({"FID": fid_value}, f)


def create_conversation(prompt):
    return [
        {
            "role": "<|User|>",
            "content": prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


distributed_state = PartialState()
accelerator = Accelerator()


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def parse_extra_args(extra_args):
    if extra_args is None or extra_args == [""]:
        return {}
    extra_args_dict = {}
    for extra_arg in extra_args:
        key, value = extra_arg.split("=")
        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        elif value[0] == "[" and value[-1] == "]":
            value = [i.replace("+", " ") for i in value[1:-1].split(",")]
            value = [v.strip() for v in value]
            if value and value[0].isdigit():
                value = [int(v) for v in value]
            elif value and value[0].replace(".", "", 1).isdigit():
                value = [float(v) for v in value]
        extra_args_dict[key] = value
    return extra_args_dict


def dummy(images, **kwargs):
    if isinstance(images, (list, tuple)) or (
            hasattr(images, "shape") and len(images.shape) > 1
    ):
        return images, [False] * images.shape[0]
    else:
        return images, False


def get_dataloader(args, num_processes=1):
    task_args = parse_extra_args(args.task_args)
    task_args["save_folder"] = args.img_save_path
    task_args["output_path"] = args.save_path

    cfg = parse_extra_args(args.generation_cfg)
    cfg = GenerationConfig(**cfg)

    dataset = Coco30kGenerationDataset(**task_args, base_cfg=cfg)
    dataloader = DataLoader(
        dataset, batch_size=num_processes, num_workers=0, shuffle=False
    )
    return dataloader


def get_evaluator(args):
    evaluator = CocoEvaluator(
        save_folder=args.img_save_path, output_path=args.save_path
    )
    return evaluator


@torch.no_grad()
def load_janus_model(model_path, lora_path=None):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    model = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)

    if lora_path:
        lora_model = PeftModel.from_pretrained(model.language_model, lora_path)

        model.language_model = lora_model.merge_and_unload()

    model = model.to(torch.bfloat16).cuda().eval()
    return vl_chat_processor, model


@torch.inference_mode()
def generate_batch(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompts: List[str],
        save_paths: List[str],
        temperature: float = 1,
        parallel_size: int = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
):
    batch_size = len(prompts)

    input_ids_list = [vl_chat_processor.tokenizer.encode(prompt) for prompt in prompts]
    max_len = max(len(ids) for ids in input_ids_list)

    tokens = torch.full((batch_size * 2, max_len), vl_chat_processor.pad_id, dtype=torch.int).cuda()
    for i in range(batch_size * 2):
        start_idx = max_len - len(input_ids_list[i // 2])
        if i % 2 == 0:
            tokens[i, start_idx:] = torch.LongTensor(input_ids_list[i // 2])
        else:
            tokens[i, start_idx] = torch.LongTensor([vl_chat_processor.tokenizer.bos_token_id])
            tokens[i, -1] = torch.LongTensor([100016])

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (batch_size, image_token_num_per_image), dtype=torch.int
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
        shape=[batch_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_imgs = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
    visual_imgs[:, :, :] = dec

    for i in range(batch_size):
        os.makedirs(os.path.dirname(save_paths[i]), exist_ok=True)
        PIL.Image.fromarray(visual_imgs[i]).save(save_paths[i])


def infer_with_janus_batch(
        dataloader: DataLoader,
        finetuned_model_path: str = None,
        base_model: str = "/path/to/huggingface/Janus-Pro-7B",
):
    processor, model = load_janus_model(base_model)
    model = model.to(distributed_state.device)

    if finetuned_model_path:
        finetuner = FineTunedModel(model.language_model.model, num_layers=5)
        finetuner.load_state_dict(torch.load(finetuned_model_path))

    print("Generating images with autoregressive model (batch mode)...")
    with distributed_state.split_between_processes(dataloader.dataset.data) as dataset:
        dataset = tqdm(dataset) if distributed_state.is_main_process else dataset
        for batch_cfgs in dataset:

            if all(Path(cfg["save_path"]).exists() for cfg in batch_cfgs):
                continue

            batch_prompts = []
            batch_save_paths = []
            for cfg in batch_cfgs:
                prompt = cfg["prompts"][0]
                if cfg["unconditional_prompt"]:
                    prompt = f"{prompt}, {cfg['unconditional_prompt']}"

                conversation = create_conversation(prompt)
                sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=processor.sft_format,
                    system_prompt="",
                )
                final_prompt = sft_format + processor.image_start_tag
                batch_prompts.append(final_prompt)
                batch_save_paths.append(cfg["save_path"])

            if finetuned_model_path:
                with finetuner:
                    generate_batch(
                        model,
                        processor,
                        batch_prompts,
                        batch_save_paths,
                    )
            else:
                generate_batch(
                    model,
                    processor,
                    batch_prompts,
                    batch_save_paths,
                )


def main(args):
    print(f"Using {distributed_state.num_processes} processes for evaluation.")
    dataloader = get_dataloader(args, num_processes=distributed_state.num_processes)

    if not args.eval_only:
        infer_with_janus_batch(
            dataloader,
            finetuned_model_path=args.ft_model_path,
            base_model=args.base_model,
        )
        accelerator.wait_for_everyone()

    if not args.gen_only and distributed_state.is_main_process:
        evaluator = get_evaluator(args)
        evaluator.evaluation()


if __name__ == "__main__":
    # model_name="official_janus_pro_7b"
    # Comment on this line of code:         default=pt_file,

    # pt_file = "/path/to/save_path/van_gogh/ft_model_ear_van_gogh.pt"
    # model_name="ear_van_gogh"

    # pt_file = "/path/to/save_path/nudity/ft_model_ear_nudity.pt"
    # model_name="ear_nudity"

    pt_file = "/path/to/save_path/church/ft_model_ear_church.pt"
    model_name = "ear_church"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_args",
        nargs="*",
        help="Extra arguments for the COCO task.",
    )
    parser.add_argument(
        "--generation_cfg",
        nargs="*",
        help="Arguments to overwrite default generation configs.",
    )
    parser.add_argument(
        "--img_save_path",
        type=str,
        default="/path/to/coco30k_fid/" + model_name + "/generated_imgs",
        help="Path to save generated images.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/path/to/coco30k_fid/" + model_name + "/eval_results_cleanfid",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/sda/home/fanhaipeng/huggingface/Janus-Pro-7B",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip generation and only evaluate the generated images.",
    )
    parser.add_argument(
        "--gen_only",
        action="store_true",
        help="Skip evaluation and only generate images.",
    )
    parser.add_argument(
        "--ft_model_path",
        default=pt_file,
        help="Path to finetuned model.",
    )

    args = parser.parse_args()
    main(args)
