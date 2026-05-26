"""
metrics.py
Known mAP / U-Recall / H-Score 계산
"""

import numpy as np
from typing import Dict, List, Tuple


# IoU

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """[xmin, ymin, xmax, ymax] 형식 두 박스의 IoU."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Pascal VOC AP (11-point interpolation)

def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """11-point interpolated AP (Pascal VOC 방식)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precisions[recalls >= thr]
        ap += (np.max(p) if p.size > 0 else 0.0)
    return ap / 11.0


# 단일 클래스 AP

def compute_ap_single_class(
    predictions: Dict[str, Dict],   # {img_id: {"known_preds": [...], ...}}
    gt_dict: Dict[str, Dict],        # {img_id: {"known": [...], "unknown": [...]}}
    class_name: str,
    iou_thr: float = 0.5,
) -> float:
    """
    특정 클래스에 대한 AP 계산.
    pred 형식: {"label": str, "bbox": [...], "score": float}
    gt  형식:  {"label": str, "bbox": [...], "difficult": bool}
    """
    # 해당 클래스 예측 전부 수집 (score 내림차순)
    all_preds = []
    for img_id, preds in predictions.items():
        for p in preds.get("known_preds", []):
            if p["label"] == class_name:
                all_preds.append((p["score"], img_id, p["bbox"]))
    all_preds.sort(key=lambda x: -x[0])

    # GT 집계
    gt_boxes: Dict[str, List] = {}
    n_positives = 0
    for img_id, gt in gt_dict.items():
        boxes = [o for o in gt["known"] if o["label"] == class_name]
        gt_boxes[img_id] = {"boxes": boxes, "matched": [False] * len(boxes)}
        n_positives += sum(1 for o in boxes if not o["difficult"])

    if n_positives == 0:
        return float("nan")

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, (score, img_id, pred_box) in enumerate(all_preds):
        gts = gt_boxes.get(img_id, {"boxes": [], "matched": []})
        best_iou, best_j = 0.0, -1
        for j, gt_obj in enumerate(gts["boxes"]):
            iou = compute_iou(pred_box, gt_obj["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0 and not gts["matched"][best_j]:
            if not gts["boxes"][best_j]["difficult"]:
                tp[i] = 1
            gts["matched"][best_j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls    = cum_tp / (n_positives + 1e-10)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)

    return voc_ap(recalls, precisions)


# Known mAP

def compute_known_map(
    predictions: Dict[str, Dict],
    gt_dict: Dict[str, Dict],
    known_classes: List[str],
    iou_thr: float = 0.5,
) -> Dict:
    """
    모든 Known 클래스에 대한 AP 계산 후 평균.
    반환: {"mAP": float, "per_class": {cls: ap}}
    """
    per_class = {}
    for cls in known_classes:
        ap = compute_ap_single_class(predictions, gt_dict, cls, iou_thr)
        per_class[cls] = ap

    valid_aps = [v for v in per_class.values() if not np.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    return {"mAP": mAP, "per_class": per_class}


# U-Recall

def compute_u_recall(
    predictions: Dict[str, Dict],
    gt_dict: Dict[str, Dict],
    iou_thr: float = 0.5,
) -> float:
    """
    Unknown GT 중 "object" 토큰 예측으로 탐지된 비율.
    pred 형식: {"bbox": [...], "score": float}  (unknown_preds)
    gt  형식:  {"bbox": [...], "difficult": bool} (unknown)
    """
    n_gt = 0
    n_recalled = 0

    for img_id, gt in gt_dict.items():
        unknown_gt = [o for o in gt["unknown"] if not o["difficult"]]
        n_gt += len(unknown_gt)

        matched_gt = [False] * len(unknown_gt)
        unknown_preds = predictions.get(img_id, {}).get("unknown_preds", [])

        for pred in unknown_preds:
            for j, gt_obj in enumerate(unknown_gt):
                if not matched_gt[j]:
                    iou = compute_iou(pred["bbox"], gt_obj["bbox"])
                    if iou >= iou_thr:
                        matched_gt[j] = True
                        break

        n_recalled += sum(matched_gt)

    return n_recalled / n_gt if n_gt > 0 else 0.0


# H-Score

def compute_h_score(mAP: float, u_recall: float) -> float:
    """Known mAP와 U-Recall의 조화평균."""
    if mAP + u_recall < 1e-10:
        return 0.0
    return 2 * mAP * u_recall / (mAP + u_recall)




def evaluate(
    predictions: Dict[str, Dict],
    gt_dict: Dict[str, Dict],
    known_classes: List[str],
    iou_thr: float = 0.5,
) -> Dict:
    """
    Known mAP + U-Recall + H-Score 한 번에 계산.
    반환: {"mAP": float, "u_recall": float, "h_score": float, "per_class": {...}}
    """
    map_result = compute_known_map(predictions, gt_dict, known_classes, iou_thr)
    u_recall   = compute_u_recall(predictions, gt_dict, iou_thr)
    h_score    = compute_h_score(map_result["mAP"], u_recall)

    return {
        "mAP":       map_result["mAP"],
        "per_class": map_result["per_class"],
        "u_recall":  u_recall,
        "h_score":   h_score,
    }