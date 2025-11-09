from accelerate.utils import tqdm
import torch
import os
from pathlib import Path

def resolve_root_dir():
    env_root = os.environ.get("ROLE_SYNTHCLIP_ROOT")
    if env_root:
        return Path(env_root)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "cpfs-data":
            return parent

    for parent in current.parents:
        if (parent / "data").exists() and (parent / "models").exists():
            return parent

    parents = list(current.parents)
    return parents[-1] if parents else current


ROOT_DIR = resolve_root_dir()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

def compute_metrics_for_multicaption(image_features, text_features,captions_per_image = 1):
    """计算 MS COCO 数据集的 Recall@K 指标，处理每个图像有5个标题的情况"""
    metrics = {}

    # 计算图像到文本和文本到图像的相似度分数
    # logits_per_image: (batch_size, batch_size)
    image_features=image_features[::captions_per_image]
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()



    # 计算 Top-K 准确率
    i2t_preds_topk = torch.topk(logits_per_image, k=max([1, 5, 10]), dim=1).indices
    t2i_preds_topk = torch.topk(logits_per_text, k=max([1, 5, 10]), dim=1).indices

    # 计算 Recall@K
    for k in [1, 5,10]:
        # 截取当前 k 值的预测
        i2t_preds = i2t_preds_topk[:, :k]
        t2i_preds = t2i_preds_topk[:, :k]
        
        # 计算 Image to Text Recall@K
        i2t_correct = []
        for i in range(len(image_features)):
            # 对于第i个图像特征，其对应的正确文本索引范围
            img_idx = i
            correct_text_indices = list(range(img_idx * captions_per_image, (img_idx + 1) * captions_per_image))
            
            # 检查预测的top-k中是否包含任何正确的文本索引
            pred_indices = i2t_preds[i].cpu().tolist()
            is_correct = any(pred_idx in correct_text_indices for pred_idx in pred_indices)
            i2t_correct.append(is_correct)
        
        # 计算 Text to Image Recall@K
        t2i_correct = []
        for t in range(len(text_features)):
            # 对于第t个文本特征，其对应的正确图像索引范围
            
            correct_img_indices = [t//captions_per_image]
            
            # 检查预测的top-k中是否包含任何正确的图像索引
            pred_indices = t2i_preds[t].cpu().tolist()
            is_correct = any(pred_idx in correct_img_indices for pred_idx in pred_indices)
            t2i_correct.append(is_correct)

        # 计算并存储指标
        metrics[f"i2t_recall@{k}"] = sum(i2t_correct) / len(i2t_correct)
        metrics[f"t2i_recall@{k}"] = sum(t2i_correct) / len(t2i_correct)
        
    return metrics
# =====================================================================================
# 3. 评估指标计算
# =====================================================================================
def compute_metrics(image_features, text_features):
    """计算 Recall@K 指标"""
    metrics = {}

    # 计算图像到文本和文本到图像的相似度分数
    # logits_per_image: (batch_size, batch_size)
    logits_per_image = image_features @ text_features.t() * temperature
    logits_per_text = logits_per_image.t()

    # 创建目标标签
    ground_truth = torch.arange(len(logits_per_image)).to(logits_per_image.device)

    # 计算 Top-K 准确率
    i2t_preds_topk = torch.topk(logits_per_image, k=max([1, 5]), dim=1).indices
    
    # 对于每个文本，找到最相似的 k 个图像的索引
    # 直接转置 logits 即可，无需重新计算 topk
    t2i_preds_topk = torch.topk(logits_per_image.t(), k=max([1, 5]), dim=1).indices

    # 计算 Recall@K
    for k in [1, 5]:
        # 截取当前 k 值的预测
        i2t_preds = i2t_preds_topk[:, :k]
        t2i_preds = t2i_preds_topk[:, :k]
        
        # 扩展 ground_truth 以便进行比较
        # .view(-1, 1) 将 ground_truth 从 (batch_size,) 变为 (batch_size, 1)
        # 这样就可以和 (batch_size, k) 的 preds 进行广播比较
        gt_expanded = ground_truth.view(-1, 1)

        # 检查真实标签是否存在于 Top-K 预测中
        i2t_correct = (i2t_preds == gt_expanded).any(dim=1)
        t2i_correct = (t2i_preds == gt_expanded).any(dim=1)

        # 计算并存储指标
        metrics[f"i2t_recall@{k}"] = i2t_correct.float().mean().item()
        metrics[f"t2i_recall@{k}"] = t2i_correct.float().mean().item()
        
    return metrics
def inference(model,dataloader,accelerator,captions_per_image=1):
    model.eval()
    all_image_features = []
    all_text_features = []
    eval_bar = tqdm(dataloader, desc="Evaluating",main_process_only=True)
    for batch in eval_bar:
        with torch.inference_mode():
            # 获取图像和文本的特征向量
            # print(type(model))
            batch.to(model.device)
            outputs = model(**batch, return_loss=True)
            image_features=outputs.image_embeds
            text_features=outputs.text_embeds

            # 使用 accelerator.gather_for_metrics 在所有设备上收集特征
            # 这比 gather 更安全，能处理不同设备上 batch size 不一致的情况
            gathered_image_features = accelerator.gather_for_metrics(image_features)
            gathered_text_features = accelerator.gather_for_metrics(text_features)
            
            all_image_features.append(gathered_image_features.detach().cpu())
            all_text_features.append(gathered_text_features.detach().cpu())
    if accelerator.is_main_process:
        final_image_features = torch.cat(all_image_features)
        final_text_features = torch.cat(all_text_features)
        
        # 特征归一化
        final_image_features /= final_image_features.norm(dim=-1, keepdim=True)
        final_text_features /= final_text_features.norm(dim=-1, keepdim=True)
        
        # --- 计算并记录评估指标 ---
        # 注意：这里的评估是在所有测试数据上一次性计算的
        # 如果测试集非常大，您可能需要分批计算然后平均指标
        eval_metrics = compute_metrics_for_multicaption(final_image_features, final_text_features,captions_per_image)
    else:
        # 非主进程时，eval_metrics必须定义，否则调用返回时会报错
        eval_metrics = None
    
    # barrier 确保所有进程同步
    accelerator.wait_for_everyone()
    
    return eval_metrics


if __name__ == "__main__":
    config = {
        "model_path": str(MODELS_DIR / "clip-vit-base-patch16"),  # 本地预训练模型路径
        "train_csv_path": str(DATA_DIR / "clip_train" / "data" / "sharedGPT4V.csv"),    # 训练数据CSV文件
        "test_csv_path": str(DATA_DIR / "clip_train" / "data" / "coco2017val_qwenvl_caption_short.csv"),      # 测试数据CSV文件
        "coco_test_csv_path": str(DATA_DIR / "clip_train" / "data" / "coco_official.csv"),
        "train_images_folder": str(DATA_DIR / "datasets" / "sharedGPT4V"),            # 图片存放的根目录
        "test_images_folder": str(DATA_DIR / "clip_train" / "images" / "val2017"),            # 图片存放的根目录
        "coco_test_image_folder": str(DATA_DIR / "clip_train" / "images" / "val2017"),
        "flickr_image_folder": str(DATA_DIR / "datasets" / "Flickr30K" / "test" / "images_flickr_1k_test"),
        "flickr_test_csv_path": str(DATA_DIR / "clip_train" / "data" / "flickr_official.csv"),
        "num_epochs": 20,                         # 训练轮次
        "batch_size": 512,                        # 批处理大小
        "learning_rate": 1e-6,                   # 学习率
        "weight_decay": 0.05,                    # 权重衰减
        "log_project_dir": "sharedGPTmixshortlongpatch16",               # TensorBoard 日志保存目录
    }

    def add_prefix_to_keys(original_dict, prefix):
        # 使用字典推导式创建一个新字典
        new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
        return new_dict
    class ImageTextDataset(Dataset):
        """自定义数据集类，用于读取图片和文本"""
        def __init__(self, csv_path: str, images_folder: str):
            """
            Args:
                csv_path (str): CSV文件的路径.
                images_folder (str): 存放图片的文件夹路径.
            """
            self.df = pd.read_csv(csv_path)
            self.images_folder = Path(images_folder)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx: int) -> dict:
            """根据索引返回一个样本，包含PIL图片对象和标题字符串"""
            row = self.df.iloc[idx]
            # 使用 .name 来确保我们只使用文件名，避免路径问题
            image_path = self.images_folder / Path(row['filepath']).name
            image = Image.open(image_path).convert("RGB")
            title = str(row['title']).replace('__comma__',',') # 确保标题是字符串
            return {"image": image, "title": title}
    