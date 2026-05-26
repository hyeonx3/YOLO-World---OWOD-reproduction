
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

# M-OWODB Task 클래스 정의 (ORE 논문 기준) 
# T1: VOC 20개 클래스
T1_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# T2: T1 + 20개 추가 (COCO 클래스 중 VOC에 없는 것)
T2_NEW_CLASSES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",
]

# T3: T2 + 20개 추가
T3_NEW_CLASSES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
]

# T4: T3 + 20개 추가
T4_NEW_CLASSES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
]

# 누적 Known 클래스 (각 Task에서 알려진 클래스 전체)
KNOWN_CLASSES: Dict[str, List[str]] = {
    "T1": T1_CLASSES,
    "T2": T1_CLASSES + T2_NEW_CLASSES,
    "T3": T1_CLASSES + T2_NEW_CLASSES + T3_NEW_CLASSES,
    "T4": T1_CLASSES + T2_NEW_CLASSES + T3_NEW_CLASSES + T4_NEW_CLASSES,
}

# Task별 새로 추가된 클래스 (per-class AP 분석용)
TASK_NEW_CLASSES: Dict[str, List[str]] = {
    "T1": T1_CLASSES,
    "T2": T2_NEW_CLASSES,
    "T3": T3_NEW_CLASSES,
    "T4": T4_NEW_CLASSES,
}

# VOC XML 라벨 → 표준 이름 매핑 (VOC와 COCO 표기 차이 정규화)
LABEL_ALIAS: Dict[str, str] = {
    "aeroplane":   "aeroplane",
    "airplane":    "aeroplane",
    "diningtable": "diningtable",
    "dining table":"diningtable",
    "motorbike":   "motorbike",
    "motorcycle":  "motorbike",
    "pottedplant": "pottedplant",
    "potted plant":"pottedplant",
    "sofa":        "sofa",
    "couch":       "sofa",
    "tvmonitor":   "tvmonitor",
    "tv":          "tvmonitor",
}

# VOC/COCO 표기 차이 통일 (ex: sofa, couch)
def normalize_label(name: str) -> str:
    name = name.strip().lower()
    return LABEL_ALIAS.get(name, name)

# test.txt에서 image_id 목록 로드
def load_image_ids(test_txt: str) -> List[str]:

    with open(test_txt) as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

# VOC XML 파싱 → 객체 목록 반환.
# 반환: [{"label": str, "bbox": [xmin, ymin, xmax, ymax], "difficult": bool},
def parse_voc_xml(xml_path: str) -> List[Dict]:
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = normalize_label(obj.find("name").text)
        difficult = obj.find("difficult")
        is_difficult = int(difficult.text) == 1 if difficult is not None else False
        bndbox = obj.find("bndbox")
        bbox = [
            float(bndbox.find("xmin").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymax").text),
        ]
        objects.append({"label": label, "bbox": bbox, "difficult": is_difficult})
    return objects


def load_gt(
    image_ids: List[str],
    annots_dir: str,
    task: str,
) -> Dict[str, Dict]:
    """
    모든 이미지의 GT를 로드하여 Known / Unknown으로 분류.

    반환:
    {
      image_id: {
        "known":   [{"label": str, "bbox": [...], "difficult": bool}, ...],
        "unknown": [{"bbox": [...], "difficult": bool}, ...],
      },
      ...
    }
    """
    known_set = set(KNOWN_CLASSES[task])
    gt_dict: Dict[str, Dict] = {}

    for img_id in image_ids:
        xml_path = os.path.join(annots_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            gt_dict[img_id] = {"known": [], "unknown": []}
            continue

        objects = parse_voc_xml(xml_path)
        known_objs   = [o for o in objects if o["label"] in known_set]
        unknown_objs = [
            {"bbox": o["bbox"], "difficult": o["difficult"]}
            for o in objects if o["label"] not in known_set
        ]
        gt_dict[img_id] = {"known": known_objs, "unknown": unknown_objs}

    return gt_dict

#known class 와 "object" token (for unknown detection) 분리 validation code

def validate_known_classes(classes: list[str]):
    FORBIDDEN = {"object"}
    contaminated = FORBIDDEN & set(classes)
    if contaminated:
        raise ValueError(f"{FORBIDDEN} in class")