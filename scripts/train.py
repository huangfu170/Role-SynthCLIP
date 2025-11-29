import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
# from tqdm.auto import tqdm
from typing import Dict, List
import deepspeed
from swanlab.integration.accelerate import SwanLabTracker
# 核心库
tracker = SwanLabTracker("Role-SynthCLIP",init_kwargs={"swanlab": {"experiment_name": "multi_positive_clip_base"}})

from datasets import load_dataset
from transformers import CLIPProcessor, get_scheduler,CLIPModel,CLIPTokenizer
from accelerate import Accelerator, notebook_launcher,ProfileKwargs
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from accelerate.utils import tqdm
from retrieval_inference import inference
from classification_inference import inference_classification
import logging
import swifter
from timm.optim.lamb import Lamb
from PIL import ImageFile
from utils import add_prefix_to_keys
from dataset import ImageTextDataset, SharedGPT4VDatset, ImageNetClassificationDataset, CIFarClassificationDataset,Food101ClassificationDataset
pd.set_option('display.max_colwidth', 10000)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =====================================================================================
# 1. 配置参数 (HYPERPARAMETERS)
# =====================================================================================
# 您可以在这里修改所有参数
import argparse
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# 1. 原始配置字典（作为默认值和后备值）
config = {
    "model_path": str(MODELS_DIR / "clip-vit-base-patch16"),
    "train_csv_path": str(DATA_DIR / "traindataset" / "MLM_filter_qwenSharedGPT4V_filter_train_checked.csv"),
    "train_images_folder": str(DATA_DIR / "traindataset" / "sharedGPT4V"),
    "coco_test_csv_path": str(DATA_DIR / "testdatasets" / "data" / "coco_official.csv"),
    "coco_test_image_folder": str(DATA_DIR / "testdatasets" / "coco2017" / "images" ),
    "flickr_image_folder": str(DATA_DIR / "testdatasets" / "Flickr30K" /  "images_flickr_1k_test"),
    "flickr_test_csv_path": str(DATA_DIR / "testdatasets" / "Flickr30K" / "flickr_official.csv"),
    "urban1k_image_folder": str(DATA_DIR / "testdatasets" / "Urban1k" / "image"),
    "urban1k_test_csv_path": str(DATA_DIR / "testdatasets" / "Urban1k" / "urban_1k.csv"),
    "num_epochs": 6,
    "batch_size": 256, # batch size per gpu, global size is batch_size * num_gpus * gradient_accumulation_steps
    "learning_rate": 1e-6,
    "weight_decay": 0.01,
    "log_project_dir": "multipositive_gather_large",
}

def parse_args_and_update_config(config_dict):
    """
    使用 argparse 解析参数，并用解析结果更新传入的配置字典。
    """
    parser = argparse.ArgumentParser(description="CLIP模型训练配置参数")

    # 路径参数 (type=str)
    parser.add_argument("--model_path", type=str, default=config_dict["model_path"],
                        help=f"本地预训练模型路径 (默认: {config_dict['model_path']})")
    parser.add_argument("--train_csv_path", type=str, default=config_dict["train_csv_path"],
                        help=f"训练数据CSV文件路径 (默认: {config_dict['train_csv_path']})")
    parser.add_argument("--train_images_folder", type=str, default=config_dict["train_images_folder"],
                        help=f"训练图片存放的根目录 (默认: {config_dict['train_images_folder']})")
    parser.add_argument("--coco_test_csv_path", type=str, default=config_dict["coco_test_csv_path"],
                        help=f"COCO测试集CSV文件路径 (默认: {config_dict['coco_test_csv_path']})")
    parser.add_argument("--coco_test_image_folder", type=str, default=config_dict["coco_test_image_folder"],
                        help=f"COCO测试图片存放的根目录 (默认: {config_dict['coco_test_image_folder']})")
    parser.add_argument("--flickr_image_folder", type=str, default=config_dict["flickr_image_folder"],
                        help=f"Flickr测试图片存放的根目录 (默认: {config_dict['flickr_image_folder']})")
    parser.add_argument("--flickr_test_csv_path", type=str, default=config_dict["flickr_test_csv_path"],
                        help=f"Flickr测试集CSV文件路径 (默认: {config_dict['flickr_test_csv_path']})")
    parser.add_argument("--urban1k_image_folder", type=str, default=config_dict["urban1k_image_folder"],
                        help=f"Urban1k测试图片存放的根目录 (默认: {config_dict['urban1k_image_folder']})")
    parser.add_argument("--urban1k_test_csv_path", type=str, default=config_dict["urban1k_test_csv_path"],
                        help=f"Urban1k测试集CSV文件路径 (默认: {config_dict['urban1k_test_csv_path']})")
    parser.add_argument("--log_project_dir", type=str, default=config_dict["log_project_dir"],
                        help=f"TensorBoard 日志保存目录 (默认: {config_dict['log_project_dir']})")

    # 数值参数
    parser.add_argument("--num_epochs", type=int, default=config_dict["num_epochs"],
                        help=f"训练轮次 (默认: {config_dict['num_epochs']})")
    parser.add_argument("--batch_size", type=int, default=config_dict["batch_size"],
                        help=f"批处理大小 (默认: {config_dict['batch_size']})")
    # 注意：学习率和权重衰减使用 float
    parser.add_argument("--learning_rate", type=float, default=config_dict["learning_rate"],
                        help=f"学习率 (默认: {config_dict['learning_rate']})")
    parser.add_argument("--weight_decay", type=float, default=config_dict["weight_decay"],
                        help=f"权重衰减 (默认: {config_dict['weight_decay']})")

    # 2. 解析参数
    args = parser.parse_args()

    # 3. 核心步骤：将 args 命名空间对象转换为字典，并用其更新 config_dict
    parsed_config = vars(args)
    config_dict.update(parsed_config)

    # 返回更新后的字典（可选，因为字典是可变对象，已被原地修改）
    return config_dict

# ----------------------------------------------------
# 4. 在程序入口调用此函数来更新全局 config 字典
parse_args_and_update_config(config)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# =====================================================================================
# 2. 数据集预处理
# =====================================================================================

import torch
import torch.nn.functional as F

def expand_positional_embedding(positional_embedding_pre, keep_len=20):
    length, dim = positional_embedding_pre.shape
    new_length = 4 * length - 3 * keep_len
    
    # 创建新的位置编码张量
    pos_emb_new = torch.zeros([new_length, dim])
    
    # 1. 保留前keep_len个位置编码
    pos_emb_new[:keep_len] = positional_embedding_pre[:keep_len]
    
    # 2. 中间部分：4倍线性插值
    for i in range(length - 1 - keep_len):
        start_idx = 4 * i + keep_len
        curr_emb = positional_embedding_pre[i + keep_len]
        next_emb = positional_embedding_pre[i + keep_len + 1]
        
        # 使用线性插值生成4个位置编码
        for j in range(4):
            alpha = j / 4  # 插值系数: 0, 0.25, 0.5, 0.75
            pos_emb_new[start_idx + j] = (1 - alpha) * curr_emb + alpha * next_emb
    
    # 3. 最后4个位置：外推
    last_emb = positional_embedding_pre[-1]
    second_last_emb = positional_embedding_pre[-2]
    diff = last_emb - second_last_emb
    
    for j in range(4):
        pos_emb_new[-4 + j] = last_emb + j * diff / 4
    
    return pos_emb_new

# 使用简化函数


def create_collate_fn(processor: CLIPProcessor):
    """
    创建一个闭包 collate_fn，它将一批数据（来自ImageTextDataset）处理成模型所需的格式。
    """
    def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: 一个列表，其中每个元素都是ImageTextDataset返回的字典。
                   例如: [{'image': PIL.Image, 'title': str}, ...]
        """
        # 从批次中分离图像和文本
        images = [item['image'] for item in batch]
        if batch[0].get('title'):
            texts = [item['title'] for item in batch]
        elif batch[0].get('label_str'):
            texts = [item['label_str'] for item in batch]
        label_idx=None
        if batch[0].get('label_idx') is not None:
            label_idx=[item['label_idx'] for item in batch]
        img_gid=None
        if batch[0].get('img_gid'):
            img_gid=[item['img_gid'] for item in batch]
        # 使用CLIPProcessor对整个批次进行处理
        # 这能确保文本被正确地填充到批次内的最大长度，并生成正确的张量形状
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",  # 填充到模型支持的最大长度 (77)
            truncation=True,
            max_length=248          # CLIP 标准文本长度
        )
        if label_idx is not None:
            inputs.update({
                'label_idx':label_idx
            })
        if img_gid is not None:
            inputs.update({
                'img_gid':img_gid
            })

        return inputs
    return collate_fn
 
        

# =====================================================================================
# 4. 主训练函数
# =====================================================================================
def main():
    # ---- 初始化 Accelerator ----
    # 使用 "tensorboard" 作为日志记录器，日志将保存在 config["log_project_dir"]


    # ---- 加载模型和处理器 ----
    try:
        processor = CLIPProcessor.from_pretrained(config["model_path"])
        tokenizer = CLIPTokenizer.from_pretrained(config["model_path"])
        model = CLIPModel.from_pretrained(config["model_path"])

        # for name,_ in model.named_parameters():
        #     print(name)
        original_pos_embed=model.text_model.embeddings.position_embedding.weight
        posisitonal_embedding_new = expand_positional_embedding(
            original_pos_embed
        )
        model.text_model.embeddings.position_embedding.weight = torch.nn.Parameter(posisitonal_embedding_new).to(model.device)
        model.config.max_position_embeddings=248
        model.config.text_config.max_position_embeddings=248
        model.text_model.embeddings.position_ids=torch.arange(model.config.max_position_embeddings).expand((1, -1))
        for param in model.parameters(): param.data = param.data.contiguous()
        
        accelerator = Accelerator(device_placement=True,log_with=tracker,project_dir=config['log_project_dir'])
        accelerator.print(f"使用设备: {accelerator.device}")
    except OSError:
        accelerator.print(f"错误: 无法从 '{config['model_path']}' 加载模型。请确认路径正确且文件完整。")
        return

    # ---- 加载和预处理数据集 ----
    try:
        train_dataset = SharedGPT4VDatset(
            csv_path=config["train_csv_path"],
            images_folder=config["train_images_folder"]
        )
        test_coco_dataset=ImageTextDataset(
            csv_path=config["coco_test_csv_path"],
            images_folder=config["coco_test_image_folder"]
        )
        test_flickr_dataset=ImageTextDataset(
            csv_path=config["flickr_test_csv_path"],
            images_folder=config["flickr_image_folder"]
        )
        test_urban_dataset=ImageTextDataset(
            csv_path=config["urban1k_test_csv_path"],
            images_folder=config["urban1k_image_folder"]
        )
        test_imagenet1k_dataset=ImageNetClassificationDataset(DATA_DIR / "testdataset" / "imagenet-val","imagenet-1k")
        test_imageneto_dataset=ImageNetClassificationDataset(DATA_DIR / "testdataset" / "imagenet-o","imagenet-o")
        test_imageneta_dataset=ImageNetClassificationDataset(DATA_DIR / "testdataset" / "imagenet-v2","imagenet-v2")
        food101_dataset=Food101ClassificationDataset(DATA_DIR / "testdataset" / "Food101" / "images")
        test_cifar10_dataset=CIFarClassificationDataset(DATA_DIR / "testdataset" / "cifar-10")
        test_cifar100_dataset=CIFarClassificationDataset(DATA_DIR / "testdataset" / "cifar-100")
        datasets = {
            "coco": test_coco_dataset,
            "flickr-1k": test_flickr_dataset,
            "urban-1k": test_urban_dataset,
            "imagenet-1k": test_imagenet1k_dataset,
            "imagenet-o": test_imageneto_dataset,
            "imagenet-v2":test_imageneta_dataset,
            "cifar10": test_cifar10_dataset,
            "cifar100": test_cifar100_dataset,
            "food-101":food101_dataset
        }
        dataloader_type = {
            "classification":[
                'cifar10',
                'cifar100',
                'imagenet-1k',
                'imagenet-o',
                'imagenet-v2',
                'food-101'
            ],
            "retrieval":[
                ('coco',5),
                ('flickr-1k',5),
                ('urban-1k',1)
            ]
        }
        dataloaders = {}
        for k,v in datasets.items():
            dataloaders[k]=DataLoader(
                v,
                shuffle=False,
                batch_size=config["batch_size"],
                collate_fn=create_collate_fn(processor),
                num_workers=1
            )
        train_dataloader=DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=config["batch_size"],
                collate_fn=create_collate_fn(processor),
                num_workers=5
            )
    except FileNotFoundError:
        accelerator.print(f"错误: 找不到CSV数据文件。请检查路径 '{config['train_csv_path']}' 和 '{config['test_csv_path']}'.")
        return
    # ---- 优化器和学习率调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    visual_params = []
    text_params = []

    for name, param in model.named_parameters():
        # print(name)
        if 'visual_projection' in name or 'vision_model' in name or 'logit_scale' in name:
            visual_params.append(param)
        elif 'text_projection' in name or 'text_model' in name:
            text_params.append(param)
    # 为每个参数组定义不同的学习率
    visual_lr = 1e-6  # 例如，视觉编码器的学习率设为1e-5
    text_lr = 1e-5   # 例如，文本编码器的学习率设为1e-4

    # 创建一个参数组列表
    optimizer_grouped_parameters = [
        {'params': visual_params, 'lr': visual_lr},
        {'params': text_params, 'lr': text_lr}
    ]
    # optimizer=Lamb(
    #     model.parameters(), 
    #     lr=config["learning_rate"],
    #     weight_decay=config["weight_decay"]
    # )
    num_training_steps = config["num_epochs"] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer, 
        num_warmup_steps=200,
        num_training_steps=num_training_steps
    )
    # lr_scheduler=deepspeed.runtime.lr_schedules.WarmupCosineLR(
    #     optimizer=optimizer, 
    #     warmup_num_steps=10,
    #     total_num_steps=num_training_steps

    # )
    print("----------------------------------------------",lr_scheduler)
    # print(f"Pre: {len(train_dataloader)}")
    # ---- 使用 `accelerator.prepare` ----
    # 这是 accelerate 的核心！它会处理模型、优化器和数据加载器的设备分配和分布式设置
    model, optimizer, train_dataloader, *prepared_dataloaders, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, *dataloaders.values(),lr_scheduler
    )
    # model, optimizer, *prepared_dataloaders, lr_scheduler = accelerator.prepare(
    #     model, optimizer, *dataloaders.values(),lr_scheduler
    # )
    i=0
    for k,v in dataloaders.items():
        dataloaders[k]=prepared_dataloaders[i]
        i+=1

    # print(f"Post: {len(train_dataloader)}")
    # accelerator.print("初始化完成")
    # ---- 初始化 TensorBoard Tracker ----
    # 只有主进程会初始化 tracker

    training_config={
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
        }
    accelerator.init_trackers(project_name="Role-SynthCLIP",config=training_config)

    # accelerator.print("tracker完成")
    # ---- 训练循环 ----
    # progress_bar = tqdm(range(num_training_steps),main_process_only=True)
    global_step = 1
    # accelerator.print("tracker完成")
    for epoch in range(config["num_epochs"]):
        model.train()
        for batch in train_dataloader:
            # 将批次数据移动到当前设备
            with accelerator.accumulate(model), accelerator.autocast():
                optimizer.zero_grad()
                if batch.get('img_gid'):
                    img_gid_local = (torch.tensor(batch.pop("img_gid"))).to(accelerator.device)
                    with torch.no_grad():
                        # 所有进程的 gid 聚合（无梯度）
                        all_img_gid = accelerator.gather(img_gid_local)  # [B]
                        # 构造正样本掩码：同一张图（同 gid）为正
                        pos_mask_i2t = all_img_gid[:, None].eq(all_img_gid[None, :])  # [B, B] bool
                        pos_counts_i2t = pos_mask_i2t.sum(dim=1, keepdim=True).clamp_min(1)  # [B,1]
                        # 正样本均分权重（每行和=1）
                        weights_i2t = pos_mask_i2t.float() / pos_counts_i2t.float()
                batch.to(accelerator.device)
                # loss= model(**batch, return_loss=True).loss
                outputs = model(**batch)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                # 4. 手动处理全局特征和梯度
                # 使用 with torch.no_grad() 来确保聚合操作不带梯度
                with torch.no_grad():
                    # 通过 accelerator.gather 获取所有 GPU 的特征，但这些特征的梯度流是断开的
                    all_image_features = accelerator.gather(image_features)
                    all_text_features = accelerator.gather(text_features)

                # 获取当前进程的索引
                rank = accelerator.local_process_index
                world_size = accelerator.num_processes

                # 从全局 tensor 中分割出每个进程的特征
                gathered_image_features_list = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features_list = list(all_text_features.chunk(world_size, dim=0))

                # 用本地有梯度的特征替换掉全局特征中无梯度的部分
                gathered_image_features_list[rank] = image_features
                gathered_text_features_list[rank] = text_features

                # 重新拼接成一个完整的全局 tensor
                all_image_features = torch.cat(gathered_image_features_list, dim=0)
                all_text_features = torch.cat(gathered_text_features_list, dim=0)

                # 5. 计算全局损失
                # 获取 CLIP 模型的 temperature 参数（通常命名为 logit_scale）
                temperature = model.module.logit_scale.exp().to(accelerator.device)

                # 计算全局相似度矩阵
                sim_i2t = torch.matmul(all_image_features, all_text_features.T) * temperature
                sim_t2i = torch.matmul(all_text_features, all_image_features.T) * temperature
                
                # 计算损失
                global_batch_size = sim_i2t.shape[0]
                log_probs_i2t = sim_i2t.log_softmax(dim=1)
                loss_i = -(weights_i2t * log_probs_i2t).sum(dim=1).mean()

                # 使用 torch.arange 创建正确的标签
                labels = torch.arange(global_batch_size, device=accelerator.device)
                # loss_i = F.cross_entropy(sim_i2t, labels)
                loss_t = F.cross_entropy(sim_t2i, labels)
                loss = (loss_i + loss_t) / 2
                # 5. 反向传播和优化
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()



            avg_loss = accelerator.gather(loss.detach()).mean().item()
            if accelerator.is_main_process:
                # 记录到 TensorBoard
                accelerator.log({
                    "train_loss": avg_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                accelerator.print(global_step)

            # progress_bar.update(1)
            global_step += 1
            accelerator.print(f"{global_step*torch.cuda.device_count()}/{num_training_steps}已完成,该轮loss为{avg_loss}")
            accelerator.print(f"lr:{lr_scheduler.get_last_lr()[0]}")
        
        accelerator.wait_for_everyone() # 等待所有进程完成

        # ---- 评估循环 ----
        classification_metric = {}
        retrieval_metric = {}
        
        for dataloader_key,caption_per_image in dataloader_type['retrieval']:
            retrieval_metric[dataloader_key]=inference(model,dataloaders[dataloader_key],accelerator,caption_per_image)
            if accelerator.is_main_process:
                eval_metric=add_prefix_to_keys(retrieval_metric[dataloader_key],dataloader_key)
                accelerator.log(eval_metric, step=global_step)
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']} | {dataloader_key} Metrics: {eval_metric}")
                
            accelerator.print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataloader_key} Metrics: {retrieval_metric[dataloader_key]}")


        for dataloader_key in dataloader_type['classification']:
            classification_metric[dataloader_key]=inference_classification(model,dataloaders[dataloader_key],accelerator,tokenizer)
            if accelerator.is_main_process:
                eval_metric=add_prefix_to_keys(classification_metric[dataloader_key],dataloader_key)
                accelerator.log(eval_metric, step=global_step)
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']} | {dataloader_key} Metrics: {eval_metric}")
            accelerator.print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataloader_key} Metrics: {classification_metric[dataloader_key]}")


        # break

    # ---- 训练结束 ----
    accelerator.wait_for_everyone() # 等待所有进程完成
    
    # ---- 保存模型 ----
    if accelerator.is_main_process:
        # 使用 unwrap_model 获取原始的 PyTorch Module
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 定义保存路径
        save_path = Path(config["log_project_dir"]) / f"epoch_{config['num_epochs']}_final"
        save_path.mkdir(exist_ok=True, parents=True)
        
        # 保存模型权重和处理器配置
        unwrapped_model.save_pretrained(save_path,save_function=accelerator.save)
        processor.save_pretrained(save_path)
        
        accelerator.print(f"训练完成！最终模型已保存至: {save_path}")

    accelerator.end_training()


if __name__ == "__main__":

    # 启动训练
    main()