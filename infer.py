"""
실행 (on GPU environment (Colab ..) ):
    python infer.py \
        --images   /content/drive/MyDrive/mowod/JPEGImages \
        --test_txt /content/drive/MyDrive/mowod/ImageSets/test.txt \
        --out_dir  /content/drive/MyDrive/mowod/results \
        --task     all          # T1 / T2 / T3 / T4 / all
"""

import os
import sys
import json
import argparse


import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from torchvision.ops import nms


# dataset.py에서 클래스 정의 가져오기
sys.path.insert(0, os.path.dirname(__file__))
from dataset import KNOWN_CLASSES, load_image_ids


# 경로 상수 (Colab 환경 기준) 
YOLO_WORLD_DIR = "/content/YOLO-World"
CONFIG_PATH    = (
    f"{YOLO_WORLD_DIR}/configs/pretrain/"
    "yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
)
WEIGHTS_PATH   = "/content/drive/MyDrive/reproduction/{YOLO_WORLD_DIR}/pretrained_weights/"

# 모델 로드

def build_runner(config_path: str, weights_path: str):
    cfg = Config.fromfile(config_path)
    cfg.work_dir = "."
    cfg.load_from = weights_path
    runner = Runner.from_cfg(cfg) #runner 객체 생성 (mmengine 제공)
    runner.call_hook("before_run")
    runner.load_or_resume() #경로에서 가중치 불러오기
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)

    runner.model.eval()
    print("Model loaded.")
    return runner


# 단일 이미지 추론 (original inference.py 의 return 수정)

def infer_single(
    runner,
    img_path: str,
    texts: list,          # [[cls1], [cls2], ..., ["object"], [""]]
    n_known: int,         # Known 클래스 수 → label_id == n_known 이면 Unknown
    score_thr: float = 0.05,
    nms_thr:   float = 0.5,
    max_boxes: int   = 300,
) -> dict:
    """
    반환:
    {
      "known_preds":   [{"label": str, "bbox": [x1,y1,x2,y2], "score": float}, ...],
      "unknown_preds": [{"bbox": [x1,y1,x2,y2], "score": float}, ...],
    }
    """
    data_info = runner.pipeline(
        dict(img_id=0, img_path=img_path, texts=texts)
    )
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred = output.pred_instances

    # NMS + score 필터
    keep = nms(pred.bboxes, pred.scores, iou_threshold=nms_thr) #NMs : iou threshold 미만의 것 버림
    pred = pred[keep]
    pred = pred[pred.scores.float() > score_thr] #신뢰도 미만의 것을 버릶

    if len(pred.scores) > max_boxes: #최대 max_boxes 개수만 남김
        idx = pred.scores.float().topk(max_boxes)[1] 
        pred = pred[idx]

    pred = pred.cpu().numpy()
    bboxes  = pred["bboxes"].tolist()
    labels  = pred["labels"].tolist()
    scores  = pred["scores"].tolist()

    # ===================================OWOD========================================================================
   
    # Known 클래스 이름 목록 (texts에서 복원)
    known_names = [t[0] for t in texts[:n_known]]

    known_preds, unknown_preds = [], []
    for bbox, label_id, score in zip(bboxes, labels, scores):
        if label_id < n_known: # Known(class)
            known_preds.append({
                "label": known_names[label_id],
                "bbox":  bbox,
                "score": score,
            })
        elif label_id == n_known:# "object" → Unknown
            unknown_preds.append({"bbox": bbox, "score": score})
        # label_id == n_known+1 은 padding [" "], 무시

    return {"known_preds": known_preds, "unknown_preds": unknown_preds}


# Task별 전체 추론 루프

def run_task(
    runner,
    task: str, # T1/T2/T3/T4
    image_ids: list,
    images_dir: str,
    out_dir: str,
    score_thr: float,
    nms_thr:   float,
):
    out_path = os.path.join(out_dir, f"preds_{task}.json")
    if os.path.exists(out_path):
        print(f"[{task}] 이미 존재: {out_path} → 스킵")
        return

    known_classes = KNOWN_CLASSES[task]
    n_known       = len(known_classes)

    # 텍스트 프롬프트: known 클래스 + "object" 토큰 + 패딩 [" "]
    texts = [[cls] for cls in known_classes] + [["object"]] + [[" "]]

    predictions = {}
    total = len(image_ids)

    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{img_id}.jpeg")
        if not os.path.exists(img_path):
            predictions[img_id] = {"known_preds": [], "unknown_preds": []}
            continue

        result = infer_single(
            runner, img_path, texts, n_known,
            score_thr=score_thr, nms_thr=nms_thr,
        )
        predictions[img_id] = result

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{task}] {i+1}/{total} 완료")

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"✅ [{task}] 저장 완료: {out_path}")


# CLI

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",    required=True, help="JPEGImages 디렉토리")
    p.add_argument("--test_txt",  required=True, help="test.txt 경로")
    p.add_argument("--out_dir",   required=True, help="결과 JSON 저장 디렉토리")
    p.add_argument("--task",      default="all", choices=["T1","T2","T3","T4","all"])
    p.add_argument("--config",    default=CONFIG_PATH)
    p.add_argument("--weights",   default=WEIGHTS_PATH)
    p.add_argument("--score_thr", type=float, default=0.05)
    p.add_argument("--nms_thr",   type=float, default=0.5)
    return p.parse_args()


def main():
    args = get_args()

    image_ids = load_image_ids(args.test_txt)
    print(f"이미지 수: {len(image_ids)}")

    runner = build_runner(args.config, args.weights)

    tasks = ["T1", "T2", "T3", "T4"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(
            runner, task, image_ids,
            images_dir=args.images,
            out_dir=args.out_dir,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
        )

    print("\nInference 완료. evaluate.py로 평가를 진행하세요.")


if __name__ == "__main__":
    main()