# python 3.6
from transformers import pipeline
import torch
from torch.nn import functional as F
import os
import json


def init_classifier(path):
    return pipeline('image-classification', model=path)


def style_eval(classifier, img):
    results = classifier(img, top_k=20)

    scores = torch.tensor([r["score"] for r in results])

    probs = F.softmax(scores, dim=0)

    for i, r in enumerate(results):
        r["score"] = probs[i].item()
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results


check_path = "/path/to/save_path/ear_van_gogh/generated_imgs/"
# download Artist classifier from paper: https://arxiv.org/abs/2310.11868
# github: https://github.com/OPTML-Group/Diffusion-MU-Attack
classifier = init_classifier('/path/to/style_classifier/checkpoint-2800')
imgs = os.listdir(check_path)

result = []
for img in imgs:
    topK = style_eval(classifier, os.path.join(check_path, img))
    result.append(topK)
import pandas as pd

data = []
for res in result:
    top_labels = [r['label'] for r in res[:10]]
    counts = [top_labels[:i + 1].count('vincent-van-gogh') for i in range(10)]
    counts.append(res[top_labels.index('vincent-van-gogh')] if sum(counts) > 0 else 'not found')
    data.append(counts)

columns = ["top-1", "top-2", "top-3", "top-4", "top-5", "top-6", "top-7", "top-8", "top-9", "top-10",
           "vincent-van-gogh"]
df = pd.DataFrame(data, columns=columns)
df.to_csv('result.csv', index=False)
