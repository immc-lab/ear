import os
import os.path as osp
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Please run the coco image generation code in code eval_fid.py first
BATCH_SIZE = 50
CLIP_MODEL = '/path/to/huggingface/clip-vit-base-patch32'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GENERATED_IMGS_DIR = "/path/to/coco30k_fid/ear_church/generated_imgs/coco30k"
# GENERATED_IMGS_DIR = "/path/to/coco30k_fid/ear_van_gogh/generated_imgs/coco30k"
# GENERATED_IMGS_DIR = "/path/to/coco30k_fid/ear_nudity/generated_imgs/coco30k"
GENERATED_IMGS_DIR = "/path/to/coco30k_fid/official_janus_pro_7b/generated_imgs/coco30k"
CSV_PATH = "../data/coco_30k.csv"


class Coco30kDataset(Dataset):
    def __init__(self, img_dir, csv_path, processor, tokenizer):
        self.img_dir = img_dir
        self.processor = processor
        self.tokenizer = tokenizer

        self.df = pd.read_csv(csv_path)
        self.prompts = self.df['prompt'].tolist()

        self.img_files = sorted([
            osp.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        assert len(self.img_files) == len(self.prompts), \
            f"The number of images ({len(self.img_files)}) does not match the number of prompts ({len(self.prompts)})"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        prompt = self.prompts[idx]

        img = Image.open(img_path)
        img = self.processor(text=None, images=img, return_tensors="pt")
        img['pixel_values'] = img['pixel_values'][0]

        text = self.tokenizer(
            prompt,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
            max_length=77
        )
        for key in text:
            text[key] = text[key].squeeze()

        return {'img': img, 'text': text}


@torch.no_grad()
def calculate_clip_score(dataloader, model):
    score_acc = 0.
    sample_num = 0.

    for batch in tqdm(dataloader, desc="Calculating CLIP Score"):
        img_features = model.get_image_features(
            pixel_values=batch['img']['pixel_values'].to(DEVICE)
        )
        text_features = model.get_text_features(
            input_ids=batch['text']['input_ids'].to(DEVICE),
            attention_mask=batch['text']['attention_mask'].to(DEVICE)
        )

        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        score = (img_features * text_features).sum()
        score_acc += score
        sample_num += img_features.shape[0]

    return score_acc / sample_num


def main():
    print("Loading CLIP model...")
    model = AutoModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)

    print("Preparing dataset...")
    dataset = Coco30kDataset(
        GENERATED_IMGS_DIR,
        CSV_PATH,
        processor,
        tokenizer
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Calculating CLIP Score...")
    clip_score = calculate_clip_score(dataloader, model)
    clip_score = clip_score.cpu().item()

    print(f"\nFinal CLIP Score: {clip_score:.4f}")


if __name__ == '__main__':
    main()
