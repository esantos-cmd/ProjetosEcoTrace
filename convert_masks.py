"""Converte máscaras coloridas (exportadas do CVAT, Label Studio etc.)
para máscaras de índice (pixel = número da classe).

Ajuste C.COLORS em seg_config.py para bater com as cores usadas na anotação.

Uso:
    python convert_masks.py --src export_cvat/SegmentationClass --dst data/train/masks
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import seg_config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tol", type=int, default=10, help="tolerância de cor")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.src).glob("*.png")):
        color = cv2.imread(str(p)).astype(np.int16)
        out = np.zeros(color.shape[:2], np.uint8)
        matched = np.zeros(color.shape[:2], bool)
        for idx, name in enumerate(C.CLASSES):
            bgr = np.array(C.COLORS[name], np.int16)
            hit = np.all(np.abs(color - bgr) <= args.tol, axis=-1)
            out[hit] = idx
            matched |= hit
        unknown = (~matched).mean() * 100
        if unknown > 0.5:
            print(f"Aviso: {p.name} tem {unknown:.1f}% de pixels com cor desconhecida (viram fundo)")
        cv2.imwrite(str(dst / p.name), out)
    print(f"Máscaras salvas em {dst}")


if __name__ == "__main__":
    main()
