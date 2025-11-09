import torch
from accelerate.utils import tqdm

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor
def generate_classnames_embedding(model,classnames,accelerator,tokenizer):
    from data.templates import imagenet_templates
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in imagenet_templates]  # format with class

            # print(type(tokenizer))
            texts = tokenizer(texts,return_tensors='pt', padding='longest').to(accelerator.device)  # tokenize
            # print(type(texts))
            class_embeddings = model.module.get_text_features(**texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            # print(class_embedding.size())
            class_embedding  /= _get_vector_norm(class_embedding)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(accelerator.device)
        return zeroshot_weights
def generate_image_features(model,dataloader,accelerator):
    all_image_features = []
    all_labels=[]
    eval_bar = tqdm(dataloader, desc="Evaluating",main_process_only=True)
    for batch in eval_bar:
        with torch.inference_mode():
            # 获取图像和文本的特征向量
            # print(type(model))
            label_idxs=torch.tensor(batch.pop('label_idx')).to(accelerator.device)
            batch.to(model.device)
            outputs = model(**batch)
            image_features=outputs['image_embeds']
            
            # 使用 accelerator.gather_for_metrics 在所有设备上收集特征
            # 这比 gather 更安全，能处理不同设备上 batch size 不一致的情况
            gathered_image_features = accelerator.gather_for_metrics(image_features)
            gathered_label_idx = accelerator.gather_for_metrics(label_idxs)
            all_image_features.append(gathered_image_features.detach())
            all_labels.append(gathered_label_idx)
    return all_image_features,all_labels
def inference_classification(model,dataloader,accelerator,tokenizer):
    model.eval()

    zeroshot_weights = generate_classnames_embedding(model,dataloader.dataset.classnames,accelerator,tokenizer)
    all_image_features,all_labels = generate_image_features(model,dataloader,accelerator)
    
    if accelerator.is_main_process:
        final_image_features = torch.cat(all_image_features)
        final_labels = torch.cat(all_labels,dim=0)    
        print(final_image_features.size(),final_labels.size())    
        # 特征归一化
        final_image_features /= final_image_features.norm(dim=-1, keepdim=True)
        
        # --- 计算并记录评估指标 ---
        # 注意：这里的评估是在所有测试数据上一次性计算的
        # 如果测试集非常大，您可能需要分批计算然后平均指标
        eval_metrics = {}
        predict = final_image_features @ zeroshot_weights
        predict = torch.topk(predict, 1, dim=1).indices
        acc = (predict.squeeze() == final_labels).sum().item() / len(predict)
        eval_metrics = {
            "acc":acc
        }
    else:
        # 非主进程时，eval_metrics必须定义，否则调用返回时会报错
        eval_metrics = None
    
    # barrier 确保所有进程同步
    accelerator.wait_for_everyone()
    
    return eval_metrics
