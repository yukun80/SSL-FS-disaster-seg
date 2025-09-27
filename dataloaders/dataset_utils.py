"""Dataset metadata for POTSDAM_BIJIE remote-sensing few-shot tasks."""

DATASET_INFO = {
    "POTSDAM_BIJIE": {
        "PSEU_LABEL_NAME": ["BGD", "SUPFG"],
        "REAL_LABEL_NAME": ["BG", "LANDSLIDE"],
        "_SEP": [0, 1, 2],
        "MODALITY": "RGB",
        "LABEL_GROUP": {
            "pa_all": {1},
            0: {1},
            1: {1},
        },
    }
}
