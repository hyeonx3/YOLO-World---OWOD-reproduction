"""
visualize.py  ←  VSCode(로컬)에서 실행
───────────────────────────────────────────────────────────────────────────────
이미지 위에 GT와 예측 박스를 겹쳐 그립니다.

색상 규칙:
  GT Known    → 초록 (solid)
  GT Unknown  → 빨강 (solid)
  Pred Known  → 파랑 (dashed)
  Pred Unknown→ 주황 (dashed)

실행:
    python visualize.py \
        --images  /path/to/mowod/JPEGImages \
        --annots  /path/to/mowod/Annotations \
        --preds   /path/to/results \
        --test_txt /path/to/mowod/ImageSets/test.txt \
        --task    T1 \
        --n       5            # 랜덤 N장 시각화
        --out_dir ./vis_output
"""

import os
import sys
import json
import argparse
import random

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import KNOWN_CLASSES, load_image_ids, load_gt


# ── 색상 (BGR) ────────────────────────────────────────────────────────────────
COLOR_GT_KNOWN    = (0,   200,   0)    # 초록
COLOR_GT_UNKNOWN  = (0,   0,   220)    # 빨강
COLOR_PRED_KNOWN  = (220, 100,   0)    # 파랑
COLOR_PRED_UNKNOWN= (0,   140, 255)    # 주황


def draw_box(img, bbox, color, label="", dashed=False, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]

    if dashed:
        # 대시 선 시뮬레이션 (OpenCV는 dashed line 미지원)
        dash_len = 10
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y1), (min(x+dash_len, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x+dash_len, x2), y2), color, thickness)
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x1, y), (x1, min(y+dash_len, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y+dash_len, y2)), color, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font_scale = 0.45
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def visualize_one(
    img_id: str,
    images_dir: str,
    gt: dict,
    preds: dict,
    task: str,
    out_dir: str,
):
    # 이미지 로드
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, f"{img_id}.jpeg")
    if not os.path.exists(img_path):
        print(f"이미지 없음: {img_id}")
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    # GT Known (초록 solid)
    for obj in gt.get("known", []):
        draw_box(img, obj["bbox"], COLOR_GT_KNOWN,
                 label=f"GT:{obj['label']}", dashed=False)

    # GT Unknown (빨강 solid)
    for obj in gt.get("unknown", []):
        draw_box(img, obj["bbox"], COLOR_GT_UNKNOWN,
                 label="GT:unknown", dashed=False)

    # Pred Known (파랑 dashed)
    for p in preds.get("known_preds", []):
        draw_box(img, p["bbox"], COLOR_PRED_KNOWN,
                 label=f"{p['label']}:{p['score']:.2f}", dashed=True)

    # Pred Unknown (주황 dashed)
    for p in preds.get("unknown_preds", []):
        draw_box(img, p["bbox"], COLOR_PRED_UNKNOWN,
                 label=f"unk:{p['score']:.2f}", dashed=True)

    # 범례
    legend = [
        ("GT Known",     COLOR_GT_KNOWN),
        ("GT Unknown",   COLOR_GT_UNKNOWN),
        ("Pred Known",   COLOR_PRED_KNOWN),
        ("Pred Unknown", COLOR_PRED_UNKNOWN),
    ]
    for i, (name, color) in enumerate(legend):
        y = 20 + i * 20
        cv2.rectangle(img, (8, y - 12), (22, y), color, -1)
        cv2.putText(img, name, (26, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{task}_{img_id}.jpg")
    cv2.imwrite(save_path, img)
    print(f"저장: {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",   required=True)
    p.add_argument("--annots",   required=True)
    p.add_argument("--preds",    required=True, help="preds_T*.json 폴더")
    p.add_argument("--test_txt", required=True)
    p.add_argument("--task",     default="T1", choices=["T1","T2","T3","T4"])
    p.add_argument("--n",        type=int, default=5, help="시각화할 이미지 수")
    p.add_argument("--img_id",   default=None,    help="특정 이미지 ID 지정")
    p.add_argument("--out_dir",  default="./vis_output")
    return p.parse_args()


def main():
    args = get_args()

    image_ids = load_image_ids(args.test_txt)
    pred_path = os.path.join(args.preds, f"preds_{args.task}.json")
    with open(pred_path) as f:
        all_preds = json.load(f)

    gt_dict = load_gt(image_ids, args.annots, args.task)

    # 시각화할 이미지 선택
    if args.img_id:
        targets = [args.img_id]
    else:
        targets = random.sample(image_ids, min(args.n, len(image_ids)))

    for img_id in targets:
        visualize_one(
            img_id,
            images_dir=args.images,
            gt=gt_dict.get(img_id, {"known": [], "unknown": []}),
            preds=all_preds.get(img_id, {"known_preds": [], "unknown_preds": []}),
            task=args.task,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()