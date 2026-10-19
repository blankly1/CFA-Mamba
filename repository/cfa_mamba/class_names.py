import os
from typing import Dict, List

import yaml
from torchvision.datasets import ImageFolder


DOMAINNET126_CLASS_TO_IDX: Dict[str, int] = {
    "aircraft_carrier": 0,
    "alarm_clock": 1,
    "ant": 2,
    "anvil": 3,
    "asparagus": 4,
    "axe": 5,
    "banana": 6,
    "basket": 7,
    "bathtub": 8,
    "bear": 9,
    "bee": 10,
    "bird": 11,
    "blackberry": 12,
    "blueberry": 13,
    "bottlecap": 14,
    "broccoli": 15,
    "bus": 16,
    "butterfly": 17,
    "cactus": 18,
    "cake": 19,
    "calculator": 20,
    "camel": 21,
    "camera": 22,
    "candle": 23,
    "cannon": 24,
    "canoe": 25,
    "carrot": 26,
    "castle": 27,
    "cat": 28,
    "ceiling_fan": 29,
    "cello": 30,
    "cell_phone": 31,
    "chair": 32,
    "chandelier": 33,
    "coffee_cup": 34,
    "compass": 35,
    "computer": 36,
    "cow": 37,
    "crab": 38,
    "crocodile": 39,
    "cruise_ship": 40,
    "dog": 41,
    "dolphin": 42,
    "dragon": 43,
    "drums": 44,
    "duck": 45,
    "dumbbell": 46,
    "elephant": 47,
    "eyeglasses": 48,
    "feather": 49,
    "fence": 50,
    "fish": 51,
    "flamingo": 52,
    "flower": 53,
    "foot": 54,
    "fork": 55,
    "frog": 56,
    "giraffe": 57,
    "goatee": 58,
    "grapes": 59,
    "guitar": 60,
    "hammer": 61,
    "helicopter": 62,
    "helmet": 63,
    "horse": 64,
    "kangaroo": 65,
    "lantern": 66,
    "laptop": 67,
    "leaf": 68,
    "lion": 69,
    "lipstick": 70,
    "lobster": 71,
    "microphone": 72,
    "monkey": 73,
    "mosquito": 74,
    "mouse": 75,
    "mug": 76,
    "mushroom": 77,
    "onion": 78,
    "panda": 79,
    "peanut": 80,
    "pear": 81,
    "peas": 82,
    "pencil": 83,
    "penguin": 84,
    "pig": 85,
    "pillow": 86,
    "pineapple": 87,
    "potato": 88,
    "power_outlet": 89,
    "purse": 90,
    "rabbit": 91,
    "raccoon": 92,
    "rhinoceros": 93,
    "rifle": 94,
    "saxophone": 95,
    "screwdriver": 96,
    "sea_turtle": 97,
    "see_saw": 98,
    "sheep": 99,
    "shoe": 100,
    "skateboard": 101,
    "snake": 102,
    "speedboat": 103,
    "spider": 104,
    "squirrel": 105,
    "strawberry": 106,
    "streetlight": 107,
    "string_bean": 108,
    "submarine": 109,
    "swan": 110,
    "table": 111,
    "teapot": 112,
    "teddy-bear": 113,
    "television": 114,
    "The_Eiffel_Tower": 115,
    "The_Great_Wall_of_China": 116,
    "tiger": 117,
    "toe": 118,
    "train": 119,
    "truck": 120,
    "umbrella": 121,
    "vase": 122,
    "watermelon": 123,
    "whale": 124,
    "zebra": 125,
}

OFFICE_CLASS_NAMES = [
    "backpack",
    "bike",
    "bike helmet",
    "bookcase",
    "bottle",
    "calculator",
    "desk chair",
    "desk lamp",
    "desktop computer",
    "file cabinet",
    "headphones",
    "keyboard",
    "laptop computer",
    "letter tray",
    "mobile phone",
    "monitor",
    "mouse",
    "mug",
    "paper notebook",
    "pen",
    "phone",
    "printer",
    "projector",
    "punchers",
    "ring binder",
    "ruler",
    "scissors",
    "speaker",
    "stapler",
    "tape dispenser",
    "trash can",
]

OFFICE_HOME_CLASS_NAMES = [
    "alarm clock",
    "backpack",
    "batteries",
    "bed",
    "bike",
    "bottle",
    "bucket",
    "calculator",
    "calendar",
    "candles",
    "chair",
    "clipboards",
    "computer",
    "couch",
    "curtains",
    "desk lamp",
    "drill",
    "eraser",
    "exit sign",
    "fan",
    "file cabinet",
    "flipflops",
    "flowers",
    "folder",
    "fork",
    "glasses",
    "hammer",
    "helmet",
    "kettle",
    "keyboard",
    "knives",
    "lamp shade",
    "laptop",
    "marker",
    "monitor",
    "mop",
    "mouse",
    "mug",
    "notebook",
    "oven",
    "pan",
    "paper clip",
    "pen",
    "pencil",
    "postit notes",
    "printer",
    "push pin",
    "radio",
    "refrigerator",
    "ruler",
    "scissors",
    "screwdriver",
    "shelf",
    "sink",
    "sneakers",
    "soda",
    "speaker",
    "spoon",
    "table",
    "telephone",
    "toothbrush",
    "toys",
    "trash can",
    "tv",
    "webcam",
]

VISDA_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
]


def _ordered_names_from_mapping(mapping: Dict[str, int]) -> List[str]:
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def _clean_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def load_class_names_file(path: str) -> List[str]:
    if path.endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return [_clean_name(name) for name in _ordered_names_from_mapping(data)]
        if isinstance(data, list):
            return [_clean_name(str(name)) for name in data]
        raise ValueError(f"Unsupported class-name yaml format: {path}")

    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return [_clean_name(name) for name in names]


def infer_class_names(config, class_names_file: str = "") -> List[str]:
    if class_names_file:
        names = load_class_names_file(class_names_file)
    elif config.dataset == "domainnet126":
        names = [_clean_name(name) for name in _ordered_names_from_mapping(DOMAINNET126_CLASS_TO_IDX)]
    elif config.dataset == "office":
        names = OFFICE_CLASS_NAMES
    elif config.dataset == "office-home":
        names = OFFICE_HOME_CLASS_NAMES
    elif config.dataset == "VISDA-C":
        names = VISDA_CLASS_NAMES
    else:
        domain_name = config.img_dataset[config.target_env]
        folder = ImageFolder(os.path.join(config.DATA.DATA_PATH, domain_name))
        names = [_clean_name(name) for name in _ordered_names_from_mapping(folder.class_to_idx)]

    if len(names) != int(config.MODEL.NUM_CLASSES):
        raise ValueError(
            f"Class-name count mismatch: got {len(names)}, "
            f"expected {config.MODEL.NUM_CLASSES}. "
            "Pass --class_names_file if this dataset uses a custom class order."
        )
    return names
