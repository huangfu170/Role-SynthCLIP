
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision.datasets import ImageFolder


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
        title = str(row['title']).replace('__comma__',', ') # 确保标题是字符串
        return {"image": image, "title": title}


class SharedGPT4VDatset(ImageTextDataset):
    def __init__(self, csv_path: str, images_folder: str):
        """
        Args:
            csv_path (str): CSV文件的路径.
            images_folder (str): 存放图片的文件夹路径.
        """
        super().__init__(csv_path, images_folder)
        def shorten(text):
            return text.split(".")[0]+'.'
        def walk_all_relpaths(root: Path):
            rels = []
            stack = [root]
            while stack:
                cur = stack.pop()
                with os.scandir(cur) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            # 生成相对路径，使用 POSIX 分隔符与 CSV 对齐
                            rel = Path(entry.path).relative_to(root).as_posix()
                            rels.append(rel)
            return set(rels)

        existing_relpaths = walk_all_relpaths(self.images_folder)
        print(self.df.iloc[1]['filepath'])
        # 2) 规范化 CSV 的 filepath（与上面的 as_posix 一致）
        self.df["filepath"] = self.df["filepath"].str.replace(str(self.images_folder) + "/", "", regex=False)


        # 3) O(1) 级别存在性判断（比逐行 exists 快很多）
        # 用 map + set.__contains__ 比 isin 稍快一点
        exists_mask = self.df["filepath"].map(existing_relpaths.__contains__)
        print(any(exists_mask))
        self.df=self.df[exists_mask].reset_index(drop=True)
        self.df['img_gid']=pd.factorize(self.df['filepath'])[0].astype('int64')
        print(f"有效图片数（去重后）: {self.df['img_gid'].nunique()}")
        # short_df=self.df.copy()
        # short_df['title']=short_df['title'].swifter.apply(shorten)
        # self.df=pd.concat([self.df,short_df],ignore_index=True)
        print(self.df.head(5))
        print(self.df.tail(5))
        print("所有训练样例{}".format(len(self.df)))
    def __getitem__(self, idx: int) -> dict:
        """根据索引返回一个样本，包含PIL图片对象和标题字符串"""
        row = self.df.iloc[idx]
        # 使用 .name 来确保我们只使用文件名，避免路径问题
        image_path = self.images_folder / Path(row['filepath'])
        image = Image.open(image_path).convert("RGB")
        title = str(row['title']).replace('__comma__',',') # 确保标题是字符串
        img_gid = int(row['img_gid'])
        return {"image": image, "title": title, "img_gid": img_gid}

class ImageNetClassificationDataset(Dataset):
    def __init__(self,image_dir,dataset_type):
        assert dataset_type in ['imagenet-1k','imagenet-o','imagenet-v2']
        if dataset_type == 'imagenet-1k':
            from data.imagenet_1k.classes import imagenet_classes
            class_txt_name = image_dir /  "classes.txt"
        elif dataset_type == 'imagenet-o':
            from data.imagenet_o.classes import imagenet_classes
            class_txt_name = image_dir /  "classes.txt"
        elif dataset_type == 'imagenet-v2': 
            from data.imagenet_v2.classes import imagenet_classes
            class_txt_name = image_dir /   "classes.txt"
        else:
            raise Exception('The dataset is not supported')
        with open(class_txt_name,'r',encoding='utf-8') as f:
            self.classes_str_mapping={}
            self.classes_idx_mapping={}
            
            i=0
            for line in f:
                folder_name, class_name = line.split(maxsplit=1)
                class_name = imagenet_classes[i]
                self.classes_str_mapping[folder_name.strip()]=class_name
                self.classes_idx_mapping[folder_name.strip()]=i
                i+=1
            self.classnames = list(self.classes_str_mapping.values())
        self.images=[]
        self.image_dir=image_dir
        walker=os.walk(image_dir)
        next(walker, None)
        for (dirpath, dirnames, filenames) in walker:
            for fm in filenames:
                self.images.append((os.path.basename(dirpath),os.path.join(dirpath,fm)))
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx:int)->dict:
        # print(self.images[idx])
        return {
            "image":Image.open(self.images[idx][1]).convert("RGB"),
            "label_str":self.classes_str_mapping[self.images[idx][0]],
            "label_idx":self.classes_idx_mapping[self.images[idx][0]]
        }

class CIFarClassificationDataset(Dataset):
    def __init__(self,image_dir,dataset_type):
        assert dataset_type in ['cifar-10','cifar-100']
        self.data = load_dataset(str(image_dir), split='test')
        if dataset_type == 'cifar-10':
            self.classnames = self.data.info.features['label'].names
            self.label_idx_key = 'label'
        elif dataset_type == 'cifar-100':
            self.classnames = self.data.info.features['fine_label'].names
            self.label_idx_key = 'fine_label'
        else: 
            pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx:int)->dict:
        # print(self.images[idx])
        label_idx = self.data[idx][self.label_idx_key]
        return {
            "image":self.data[idx]['img'].convert("RGB"),
            "label_str":self.classnames[label_idx],
            "label_idx":label_idx
        }

class Food101ClassificationDataset(Dataset):
    def __init__(self, image_dir):
        self.images = []
        self.image_dir = image_dir
        self.classnames = []
        self.classes_str_mapping = {}
        self.classes_idx_mapping = {}
        
        # 移除 next(walker, None) 这一行！
        walker = os.walk(image_dir)
        # 收集所有类别名
        class_names = set()
        for (dirpath, dirnames, filenames) in walker:
            for dirname in dirnames:
                clean_class_name = os.path.basename(dirname).strip()
                class_names.add(clean_class_name)
        
        # 建立类别映射
        sorted_class_names = sorted(list(class_names))
        # print(sorted_class_names)
        for i, class_name in enumerate(sorted_class_names):
            self.classes_idx_mapping[class_name] = i
            self.classnames.append(class_name.replace('_',' '))
            self.classes_str_mapping[i] = class_name
        
        # 重新遍历收集图像
        for (dirpath, dirnames, filenames) in os.walk(image_dir):
            current_class = os.path.basename(dirpath).strip()
            if current_class in self.classes_idx_mapping:
                for fm in filenames:
                    # 只处理图像文件
                    if fm.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.images.append((current_class, os.path.join(dirpath, fm)))
        # print(self.classes_idx_mapping.items())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        class_name, image_path = self.images[idx]
        label_idx = self.classes_idx_mapping[class_name]
        # print(label_idx)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            # 返回默认图像避免崩溃
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        return {
            "image": image,
            "label_str": class_name,
            "label_idx": label_idx
        }

if __name__ == '__main__':
    # ImageNetClassificationDataset(DATA_DIR / "datasets" / "imagenet-val",'imagenet-1k')
    dataset = Food101ClassificationDataset(str("data" / "testdataset" / "Food101"))
    print(dataset.classnames)
    print(dataset[12])