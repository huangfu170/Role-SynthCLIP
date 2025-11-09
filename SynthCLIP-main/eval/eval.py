import sys
import os
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accelerate import Accelerator
from typing import List,Dict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageTextDataset,ImageNetClassificationDataset,CIFarClassificationDataset,Food101ClassificationDataset
from Training.tokenizer import SimpleTokenizer
from Training.models import CLIP_VITB16
from classification_inference import inference_classification
from retrieval_inference import inference
config = {
    "model_path": "/mnt/cpfs-data/models/clip-vit-large-patch14",
    "coco_test_csv_path": "/mnt/cpfs-data/scripts/train/clip_train/data/coco_official.csv",
    "train_images_folder": "/mnt/cpfs-data/datasets/sharedGPT4V",
    "test_images_folder": "/mnt/cpfs-data/scripts/train/clip_train/images/val2017",
    "coco_test_image_folder": "/mnt/cpfs-data/datasets/coco2017/val2017",
    "flickr_image_folder": "/mnt/cpfs-data/datasets/Flickr30K/test/images_flickr_1k_test",
    "flickr_test_csv_path": "/mnt/cpfs-data//scripts/train/clip_train/data/flickr_official.csv",
    "urban1k_image_folder": "/mnt/cpfs-data/datasets/Urban1k/image",
    "urban1k_test_csv_path": "/mnt/cpfs-data/scripts/train/clip_train/data/urban_1k.csv",
    "batch_size":100
}
def add_prefix_to_keys(original_dict, prefix):
    # 使用字典推导式创建一个新字典
    new_dict = {f"{prefix}{key}": value for key, value in original_dict.items()}
    return new_dict
def create_collate_fn(tokenizer,transforms):
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
        # 使用CLIPProcessor对整个批次进行处理
        # 这能确保文本被正确地填充到批次内的最大长度，并生成正确的张量形状
        inputs={}
        inputs.update(tokenizer(texts, padding=True, truncation=True, max_length=248, return_tensors="pt"))
        inputs.update({
            'image': [transforms(image) for image in images]
        })
        if label_idx is not None:
            inputs.update({
                'label_idx':label_idx
            })

        return inputs
    return collate_fn

tokenizer=SimpleTokenizer()
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
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
test_imagenet1k_dataset=ImageNetClassificationDataset('/mnt/cpfs-data/datasets/imagenet-val','imagenet-1k')
test_imageneto_dataset=ImageNetClassificationDataset('/mnt/cpfs-data/datasets/imagenet-o','imagenet-o')
test_imageneta_dataset=ImageNetClassificationDataset('/mnt/cpfs-data/datasets/imagenetv2','imagenet-v2')
food101_dataset=Food101ClassificationDataset('/mnt/cpfs-data/datasets/Food101/images')
test_cifar10_dataset=CIFarClassificationDataset('cifar-10')
test_cifar100_dataset=CIFarClassificationDataset('cifar-100')
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
        collate_fn=create_collate_fn(tokenizer,transform),
        num_workers=2
    )

model=CLIP_VITB16()
checkpoint_path = "/mnt/cpfs-data/scripts/train/clip_train/baselines/SynthCLIP-main/checkpoint/checkpoint_best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
new_checkpoint=OrderedDict()
prefix_length=len('module.')
for key,value in checkpoint['state_dict'].items():
    new_key=key[prefix_length:]
    new_checkpoint[new_key]=value
load_status = model.load_state_dict(new_checkpoint)
accelerator=Accelerator()
model,*prepared_dataloaders=accelerator.prepare(model,*dataloaders.values())
i=0
for k,v in dataloaders.items():
    dataloaders[k]=prepared_dataloaders[i]
    i+=1
model.eval()
model.to("cuda:0")
classification_metric={}
retrieval_metric={}
global_step=0
epoch=0
with torch.no_grad():
    for dataset_key,caption_per_image in dataloader_type['retrieval']:
        retrieval_metric[dataset_key]=inference(model,dataloaders[dataset_key],accelerator,caption_per_image)
        if accelerator.is_main_process:
            eval_metric=add_prefix_to_keys(retrieval_metric[dataset_key],dataset_key)
            accelerator.log(eval_metric, step=global_step)
            print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataset_key} Metrics: {eval_metric}")
        print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataset_key} Metrics: {retrieval_metric[dataset_key]}")
    for dataset_key in dataloader_type['classification']:
        classification_metric[dataset_key]=inference_classification(model,dataloaders[dataset_key],accelerator,tokenizer)
        if accelerator.is_main_process:
            eval_metric=add_prefix_to_keys(classification_metric[dataset_key],dataset_key)
            accelerator.log(eval_metric, step=global_step)
            print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataset_key} Metrics: {eval_metric}")
        print(f"Epoch {epoch+1}/{config['num_epochs']} | {dataset_key} Metrics: {classification_metric[dataset_key]}")

