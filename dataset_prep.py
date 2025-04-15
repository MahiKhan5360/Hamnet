import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BiomedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # Validate pairs
        self.valid_pairs = []
        for img_name in self.images:
            mask_name = img_name.replace('.jpg', '_segmentation.png')
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                self.valid_pairs.append((img_name, mask_name))
            else:
                print(f"Warning: Skipping {img_name} - mask {mask_name} not found")

        if not self.valid_pairs:
            raise ValueError(f"No valid image-mask pairs in {image_dir}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # Resize both
            resize = transforms.Resize((self.image_size, self.image_size),
                                    interpolation=transforms.InterpolationMode.BILINEAR)
            image = resize(image)
            mask = resize(mask)

            if self.transform:
                image = self.transform(image)
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()

            return image, mask
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            return (torch.zeros(3, self.image_size, self.image_size),
                    torch.zeros(1, self.image_size, self.image_size))

def get_transforms(image_size=256):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=8, image_size=256):
    train_transform, val_transform = get_transforms(image_size)

    train_img_dir = os.path.join(data_dir, 'train')
    train_mask_dir = os.path.join(data_dir, 'train_ground_truth')
    val_img_dir = os.path.join(data_dir, 'validation')
    val_mask_dir = os.path.join(data_dir, 'validation_ground_truth')
    test_img_dir = os.path.join(data_dir, 'test')
    test_mask_dir = os.path.join(data_dir, 'test_ground_truth')

    train_dataset = BiomedicalImageDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        image_size=image_size
    )

    val_dataset = BiomedicalImageDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        image_size=image_size
    )

    test_dataset = BiomedicalImageDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        transform=val_transform,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader
