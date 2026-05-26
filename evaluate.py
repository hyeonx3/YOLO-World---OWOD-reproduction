"""
evaluate.py  ←  VSCode(로컬)에서 실행
───────────────────────────────────────────────────────────────────────────────
preds_T1~T4.json + VOC XML GT → mAP / U-Recall / H-Score 출력

실행:
    python evaluate.py \
        --preds   /path/to/results \
        --annots  /path/to/mowod/Annotations \
        --test_txt /path/to/mowod/ImageSets/test.txt \
        --task    all        # T1 / T2 / T3 / T4 / all
        --iou_thr 0.5
        --save_csv           # summary.csv 저장
"""

import os
import sys
import json
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import KNOWN_CLASSES, TASK_NEW_CLASSES, load_image_ids, load_gt
from metrics import evaluate


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_result(task: str, result: dict, known_classes: list):
    print(f"\n{'='*52}")
    print(f"  Task: {task}  |  Known: {len(known_classes)} classes")
    print(f"{'='*52}")
    print(f"  Known mAP : {result['mAP']     * 100:.2f} %")
    print(f"  U-Recall  : {result['u_recall'] * 100:.2f} %")
    print(f"  H-Score   : {result['h_score']  * 100:.2f} %")

    print(f"\n  Per-class AP (new classes in {task}):")
    for cls in TASK_NEW_CLASSES[task]:
        ap  = result["per_class"].get(cls, float("nan"))
        tag = f"{ap*100:.2f}%" if not np.isnan(ap) else "N/A (GT 없음)"
        print(f"    {cls:<24} {tag}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds",    required=True, help="preds_T*.json 폴더 경로")
    p.add_argument("--annots",   required=True, help="Annotations 디렉토리")
    p.add_argument("--test_txt", required=True, help="test.txt 경로")
    p.add_argument("--task",     default="all", choices=["T1","T2","T3","T4","all"])
    p.add_argument("--iou_thr",  type=float, default=0.5)
    p.add_argument("--save_csv", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    image_ids = load_image_ids(args.test_txt)
    print(f"이미지 수: {len(image_ids)}")

    tasks = ["T1", "T2", "T3", "T4"] if args.task == "all" else [args.task]
    summary_rows = []

    for task in tasks:
        pred_path = os.path.join(args.preds, f"preds_{task}.json")
        if not os.path.exists(pred_path):
            print(f"[{task}] 예측 파일 없음: {pred_path} → 스킵")
            continue

        with open(pred_path) as f:
            predictions = json.load(f)

        known_classes = KNOWN_CLASSES[task]
        gt_dict = load_gt(image_ids, args.annots, task)

        result = evaluate(predictions, gt_dict, known_classes, args.iou_thr)
        print_result(task, result, known_classes)

        summary_rows.append({
            "Task":          task,
            "Known (#cls)":  len(known_classes),
            "Known mAP (%)": round(result["mAP"]     * 100, 2),
            "U-Recall (%)":  round(result["u_recall"] * 100, 2),
            "H-Score (%)":   round(result["h_score"]  * 100, 2),
        })

    # 요약 테이블
    if len(summary_rows) > 1:
        print(f"\n{'='*52}")
        print("  Summary")
        print(f"{'='*52}")
        print(f"{'Task':<6} {'#Known':<8} {'mAP':>8} {'U-Rec':>8} {'H':>8}")
        print("-" * 52)
        for row in summary_rows:
            print(
                f"{row['Task']:<6} {row['Known (#cls)']:<8}"
                f"{row['Known mAP (%)']:>7.2f}% "
                f"{row['U-Recall (%)']:>7.2f}% "
                f"{row['H-Score (%)']:>7.2f}%"
            )

    # CSV 저장
    if args.save_csv and summary_rows:
        csv_path = os.path.join(args.preds, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nCSV 저장 완료: {csv_path}")


if __name__ == "__main__":
    main()