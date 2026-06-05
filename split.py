"""
Ham WAV dosyalarini train / validation / test klasorlerine bol.
Kullanim: python split.py --source raw_data --ratio 0.8 0.1 0.1
"""
import argparse
import os
import random
import shutil

CLASSES = ["normal", "abnormal"]
SPLITS = ["train", "validation", "test"]


def split_dataset(source_dir, dataset_dir, train_ratio, val_ratio, test_ratio, seed=42):
    ratios = [train_ratio, val_ratio, test_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Oranlarin toplami 1 olmali")

    random.seed(seed)

    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

    for cls in CLASSES:
        src_cls = os.path.join(source_dir, cls)
        if not os.path.isdir(src_cls):
            raise FileNotFoundError(f"Klasor bulunamadi: {src_cls}")

        files = [f for f in os.listdir(src_cls) if f.lower().endswith(".wav")]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        buckets = {
            "train": files[:n_train],
            "validation": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split, names in buckets.items():
            dest = os.path.join(dataset_dir, split, cls)
            for name in names:
                shutil.copy2(
                    os.path.join(src_cls, name),
                    os.path.join(dest, name),
                )

        print(f"{cls}: train={len(buckets['train'])}, val={len(buckets['validation'])}, test={len(buckets['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="raw_data", help="normal/ ve abnormal/ alt klasorleri olan kaynak")
    parser.add_argument("--out", default="dataset", help="Hedef dataset klasoru")
    parser.add_argument("--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.source, args.out, *args.ratio, seed=args.seed)
    print("Bolme tamamlandi.")
