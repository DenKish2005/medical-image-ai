"""Manifest-based CXR loading with strict patient-level leakage checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")
UNCERTAINTY_POLICIES = ("ignore", "zero", "one")


def read_manifest(path: str | Path, diseases: list[str], uncertainty_policy: str):
    if uncertainty_policy not in UNCERTAINTY_POLICIES:
        raise ValueError(f"Explicit uncertainty_policy required: {UNCERTAINTY_POLICIES}")
    df = pd.read_csv(path, dtype={"patient_id": "string", "path": "string", "split": "string"})
    mandatory = {"patient_id", "path", "split", *diseases}
    if missing := mandatory.difference(df.columns):
        raise ValueError(f"Manifest missing columns {sorted(missing)}")
    if df.empty or df[["patient_id","path","split"]].isna().any().any():
        raise ValueError("Empty manifest or missing patient/path/split")
    if not df["split"].isin(SPLITS).all():
        raise ValueError("Only train/val/test splits allowed")
    if df.duplicated(subset=["split", "path"]).any():
        raise ValueError("Duplicate image path within split")
    patient_splits = df.groupby("patient_id")["split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError("PATIENT LEAKAGE: patient_id appears in multiple splits")
    for disease in diseases:
        values = pd.to_numeric(df[disease], errors="coerce")
        if not values.dropna().isin((-1,0,1)).all():
            raise ValueError(f"Disease '{disease}' must contain -1,0,1 or blank")
        df[disease] = values
    return df


def label_arrays(df: pd.DataFrame, diseases: list[str], policy: str):
    raw = df[diseases].to_numpy(dtype=np.float32)
    mask = np.isfinite(raw) & (raw != -1 if policy == "ignore" else True)
    target = np.nan_to_num(raw, nan=0)
    if policy == "one":
        target[target == -1] = 1
    else:
        target[target == -1] = 0
    return target.astype(np.float32), mask.astype(np.bool_)


def patient_manifest_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CXRDataset(Dataset):
    def __init__(self, manifest, *, split, root, diseases, uncertainty_policy,
                 image_size=224, use_roi=False):
        df = read_manifest(manifest, diseases, uncertainty_policy)
        self.df = df.loc[df.split == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No examples for split '{split}'")
        self.root = Path(root)
        self.diseases = diseases
        self.targets, self.masks = label_arrays(self.df, diseases, uncertainty_policy)
        self.image_size = image_size
        self.use_roi = use_roi and split == "train"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.root / str(row["path"])
        with Image.open(image_path) as im:
            gray = im.convert("L").resize((self.image_size, self.image_size))
            values = np.asarray(gray, dtype=np.float32).copy()/255.0
        x = torch.from_numpy(np.stack([values]*3))
        x = (x-torch.tensor([0.485,0.456,0.406])[:,None,None])/torch.tensor([0.229,0.224,0.225])[:,None,None]
        # Boxes are normalized [x0,y0,x1,y1] and only loaded on train.
        boxes = []
        if self.use_roi and "roi_boxes" in self.df.columns and pd.notna(row.get("roi_boxes")):
            boxes = json.loads(row["roi_boxes"])
            for box in boxes:
                if len(box)!=4 or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
                    raise ValueError("ROI boxes must be normalized x0,y0,x1,y1")
        return {"image":x, "label":torch.tensor(self.targets[idx]),
                "valid":torch.tensor(self.masks[idx]), "patient_id":str(row.patient_id),
                "path":str(row["path"]), "boxes":boxes}


def cxr_collate(batch):
    return {"image":torch.stack([b["image"] for b in batch]),
            "label":torch.stack([b["label"] for b in batch]),
            "valid":torch.stack([b["valid"] for b in batch]),
            "patient_id":[b["patient_id"] for b in batch],
            "path":[b["path"] for b in batch], "boxes":[b["boxes"] for b in batch]}


def boxes_to_mask(boxes: list[list], height: int, width: int, device):
    m = torch.zeros((len(boxes),height,width), device=device)
    for i, per_image in enumerate(boxes):
        for x0,y0,x1,y1 in per_image:
            r0,r1 = int(y0*height), max(int(np.ceil(y1*height)),int(y0*height)+1)
            c0,c1 = int(x0*width), max(int(np.ceil(x1*width)),int(x0*width)+1)
            m[i,r0:r1,c0:c1] = 1
    return m
