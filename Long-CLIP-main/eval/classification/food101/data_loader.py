import os

import torch
from torchvision import datasets


class Food101Category():
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __init__(self):
        pass

    def __call__(self, full_path):
        img_name = full_path.split("/")[-1]
        category = full_path.split("/")[-2]
        return category


class Food101Dataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, *args, **kwargs):
        super(Food101Dataset, self).__init__(*args, **kwargs)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        sample, target = super(Food101Dataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        new_target = Food101Category()(path)
        original_tuple = (sample, new_target)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def get_label(fold_name):
    from classes import food101_classes
    return torch.tensor([food101_classes.index(fold_name)])


def data_loader(transform, args):
    food101_data = Food101Dataset(args.data_dir, transform)
    data_loader = torch.utils.data.DataLoader(
        food101_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return data_loader, food101_data

if __name__ == "__main__":
    print(get_label("n03584254"))