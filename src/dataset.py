"""Încărcarea setului de date: Dataset, DataLoader, transformări și împărțirea train/val/test (70/15/15)."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["good_asphalt", "potholes", "cracks", "unpaved_gravel"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# statistici ImageNet
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def scan_dataset(root):
    """Returnează lista (cale_imagine, eticheta) pentru toate imaginile din `root/<clasa>/`."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folderul cu date nu există: {root}")

    samples = []
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Lipsește folderul clasei '{class_name}': {class_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
        if not files:
            raise RuntimeError(f"Nu există imagini în {class_dir}")
        samples.extend((str(p.relative_to(root)), CLASS_TO_IDX[class_name]) for p in files)
    return samples


def split_dataset(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Împărțire stratificată în train / val / test (implicit 70 / 15 / 15)."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    p_rest, p_test, y_rest, y_test = train_test_split(
        paths, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )
    val_rel = val_ratio / (1.0 - test_ratio)
    p_train, p_val, y_train, y_val = train_test_split(
        p_rest, y_rest, test_size=val_rel, stratify=y_rest, random_state=seed
    )
    return {
        "train": list(zip(p_train, y_train)),
        "val": list(zip(p_val, y_val)),
        "test": list(zip(p_test, y_test)),
    }


def load_or_create_splits(root, split_file=None, seed=42, overwrite=False):
    root = Path(root)
    split_file = Path(split_file) if split_file else root.parent / "splits.json"

    if split_file.exists() and not overwrite:
        with open(split_file, encoding="utf-8") as f:
            data = json.load(f)
        return {k: [tuple(s) for s in v] for k, v in data["splits"].items()}

    splits = split_dataset(scan_dataset(root), seed=seed)

    sets = {k: {p for p, _ in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])

    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "seed": seed, "splits": splits}, f, indent=1)
    return splits


def get_transforms(img_size=224):
    """Transformări pentru antrenare (cu augmentare) și pentru evaluare (deterministe)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),        
        transforms.RandomRotation(10),            
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class RoadSurfaceDataset(Dataset):
    """Dataset PyTorch pentru imaginile de carosabil."""

    def __init__(self, root, samples, transform=None):
        self.root = Path(root)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = Counter(label for _, label in self.samples)
        return [counts.get(i, 0) for i in range(len(CLASSES))]


def get_dataloaders(root=DEFAULT_DATA_DIR, batch_size=32, img_size=224, num_workers=2, seed=42):
    """Construiește DataLoader-ele pentru train / val / test."""
    splits = load_or_create_splits(root, seed=seed)
    train_tf, eval_tf = get_transforms(img_size)

    datasets = {
        "train": RoadSurfaceDataset(root, splits["train"], train_tf),
        "val": RoadSurfaceDataset(root, splits["val"], eval_tf),
        "test": RoadSurfaceDataset(root, splits["test"], eval_tf),
    }
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin,
            generator=g if name == "train" else None,
        )
        for name, ds in datasets.items()
    }
    return loaders, datasets


def _denormalize(x):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verificarea Dataset-ului și a DataLoader-ului")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-grid", default="results/sample_batch.png")
    args = parser.parse_args()

    loaders, datasets = get_dataloaders(args.data_dir, args.batch_size, args.img_size, args.num_workers)

    total = sum(len(ds) for ds in datasets.values())
    print(f"Clase: {CLASSES}")
    print(f"Total imagini: {total}")
    for name, ds in datasets.items():
        counts = ", ".join(f"{c}={n}" for c, n in zip(CLASSES, ds.class_counts()))
        print(f"  {name:5s}: {len(ds):5d} ({100 * len(ds) / total:4.1f}%)  [{counts}]")

    images, labels = next(iter(loaders["train"]))
    print(f"Batch train: imagini {tuple(images.shape)} {images.dtype}, etichete {tuple(labels.shape)}")
    print(f"Valori după normalizare: min={images.min():.2f}, max={images.max():.2f}, medie={images.mean():.2f}")

    if args.save_grid:
        from torchvision.utils import save_image

        out = Path(args.save_grid)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_image(_denormalize(images), out, nrow=4)
        print(f"Exemplu de batch (augmentat) salvat în: {out}")


if __name__ == "__main__":
    main()
