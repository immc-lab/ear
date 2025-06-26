# 导入必要的库
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os, argparse
import torch
import pandas as pd

def check_church_in_topk(categories_list, topk):
    """检查前topk个分类结果中是否包含church"""
    return ['yes' if 'church' in [cat.lower() for cat in cats[:topk]] else 'no' 
            for cats in categories_list]

if __name__=='__main__':
    # 1. 参数解析
    parser = argparse.ArgumentParser(
                    prog = 'ObjectImageClassification',
                    description = 'Classify images and detect church presence')
    parser.add_argument('--folder_path', help='path to images', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--topk', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=250)
    
    args = parser.parse_args()

    # 2. 初始化设置
    folder = args.folder_path
    topk = args.topk
    device = args.device
    batch_size = args.batch_size
    save_path = args.save_path or f'{folder}/classification_results.csv'
    
    # 3. 模型加载
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    # 4. 准备数据
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 5. 初始化结果存储
    results = {
        'file_name': image_files,
    }
    
    # 为每个topk结果创建存储列表
    for k in range(1, topk + 1):
        results[f'category_top{k}'] = []
        results[f'score_top{k}'] = []
    
    # 6. 批处理推理
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        
        # 处理当前批次的图像
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
            
        # 模型推理
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            predictions = model(batch_tensor).softmax(1)
        
        # 获取topk结果
        probs, class_ids = torch.topk(predictions, topk, dim=1)
        
        # 存储结果
        for k in range(1, topk + 1):
            batch_categories = [weights.meta["categories"][idx] for idx in class_ids[:, k-1].cpu().numpy()]
            results[f'category_top{k}'].extend(batch_categories)
            results[f'score_top{k}'].extend(probs[:, k-1].cpu().numpy())

    # 7. 添加church检测结果
    for k in range(1, topk + 1):
        categories_list = [results[f'category_top{i}'] for i in range(1, k + 1)]
        categories_list = list(zip(*categories_list))
        results[f'has_church_top{k}'] = check_church_in_topk(categories_list, k)

    # 8. 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}") 
# python object_img_classify_0530.py --folder_path /sda/data/fanhaipeng/EJP/SixCD/model_test_results/all_images --topk 10 --batch_size 250