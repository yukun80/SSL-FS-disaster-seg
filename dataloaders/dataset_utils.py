"""Dataset metadata for remote-sensing few-shot tasks."""

from __future__ import annotations

from typing import Dict, List


def _build_class_id_map(labels: List[str]) -> Dict[str, int]:
    """Create a {class_name: class_id} map while keeping indices stable."""

    return {name: idx for idx, name in enumerate(labels)}


DATASET_INFO = {
    "POTSDAM_BIJIE": {
        "PSEU_LABEL_NAME": ["BGD", "SUPFG"],
        "REAL_LABEL_NAME": ["BG", "LANDSLIDE"],
        "CLASS_ID_MAP": _build_class_id_map(["BG", "LANDSLIDE"]),
        "BACKGROUND_ID": 0,
        "DEFAULT_ACT_LABEL_IDS": [1],
        "FORBIDDEN_CLASS_IDS": [],
        "_SEP": [0, 1, 2],
        "MODALITY": "RGB",
        "LABEL_GROUP": {
            "pa_all": {1},
            0: {1},
            1: {1},
        },
    },
    "POTSDAM_OPENEARTHMAP": {
        "PSEU_LABEL_NAME": [
            "BGD",
            "Bareland",
            "Rangeland",
            "Developed",
            "Road",
            "Tree",
            "Water",
            "Agricultural",
            "Building",
        ],
        "REAL_LABEL_NAME": [
            "BG",
            "Bareland",
            "Rangeland",
            "Developed",
            "Road",
            "Tree",
            "Water",
            "Agricultural",
            "Building",
        ],
        "CLASS_ID_MAP": _build_class_id_map(
            [
                "BG",
                "Bareland",
                "Rangeland",
                "Developed",
                "Road",
                "Tree",
                "Water",
                "Agricultural",
                "Building",
            ]
        ),
        "BACKGROUND_ID": 0,
        "DEFAULT_ACT_LABEL_IDS": list(range(1, 9)),
        "FORBIDDEN_CLASS_IDS": [],
        "MODALITY": "RGB",
    },
}
