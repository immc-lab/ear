from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os, argparse
import torch
import pandas as pd


def check_church_in_topk(categories_list, topk):
    return ['yes' if 'church' in [cat.lower() for cat in cats[:topk]] else 'no'
            for cats in categories_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ObjectImageClassification',
        description='Classify images and detect church presence')
    parser.add_argument('--folder_path', help='path to images', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--topk', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=250)

    args = parser.parse_args()

    folder = args.folder_path
    topk = args.topk
    device = args.device
    batch_size = args.batch_size
    save_path = args.save_path or f'{folder}/classification_results.csv'

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = {
        'file_name': image_files,
    }

    for k in range(1, topk + 1):
        results[f'category_top{k}'] = []
        results[f'score_top{k}'] = []

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []

        for img_file in batch_files:
            img_path = os.path.join(folder, img_file)
            try:
                img = Image.open(img_path)
                batch_images.append(preprocess(img))
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            predictions = model(batch_tensor).softmax(1)

        probs, class_ids = torch.topk(predictions, topk, dim=1)

        for k in range(1, topk + 1):
            batch_categories = [weights.meta["categories"][idx] for idx in class_ids[:, k - 1].cpu().numpy()]
            results[f'category_top{k}'].extend(batch_categories)
            results[f'score_top{k}'].extend(probs[:, k - 1].cpu().numpy())

    for k in range(1, topk + 1):
        categories_list = [results[f'category_top{i}'] for i in range(1, k + 1)]
        categories_list = list(zip(*categories_list))
        results[f'has_church_top{k}'] = check_church_in_topk(categories_list, k)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
