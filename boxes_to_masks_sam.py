"""Gera máscaras de segmentação a partir das CAIXAS YOLO usando o SAM.

Cada caixa do .txt vira um "prompt" para o SAM (Segment Anything), que recorta
o contorno real da peça dentro da caixa. Opcionalmente, separa a gordura
(pixels claros/esbranquiçados dentro da peça) por cor.

Valores na máscara (iguais ao seg_config.CLASSES):
    0 = fundo, 1 = carne_magra, 2 = gordura, 3 = osso

Estrutura:
    data/train/images/foto.jpg
    data/train/labels/foto.txt    <- caixas YOLO (classe xc yc w h)
    data/train/masks/foto.png     <- gerado aqui
    data/train/preview/foto.jpg   <- (opcional) para você conferir

Uso (dentro da pasta Segmentacao):
    python boxes_to_masks_sam.py --split data/train --gordura --preview
    python boxes_to_masks_sam.py --split data/val   --gordura --preview

Na primeira execução o ultralytics baixa o modelo do SAM automaticamente.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM

from dataset import IMG_EXTS
import seg_config as C

CARNE, GORDURA = 1, 2


def ler_caixas(txt: Path, w: int, h: int):
    """Retorna lista de caixas [x1, y1, x2, y2] em pixels e lista de polígonos (se houver)."""
    caixas, poligonos = [], []
    for line in txt.read_text().splitlines():
        vals = line.split()
        if len(vals) == 5:
            xc, yc, bw, bh = map(float, vals[1:])
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            caixas.append([max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)])
        elif len(vals) >= 7:  # já é polígono: usa direto
            pts = np.array(vals[1:], np.float32).reshape(-1, 2) * [w, h]
            poligonos.append(pts.round().astype(np.int32))
    return caixas, poligonos


def preencher_buracos(m: np.ndarray) -> np.ndarray:
    """Preenche buracos internos (ex.: gordura que o SAM deixou de fora) e fica só com a maior região."""
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        cv2.drawContours(out, [maior], -1, 1, thickness=cv2.FILLED)
    return out


def detectar_gordura(img_bgr: np.ndarray, peca: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    """Pixels claros e pouco saturados dentro da peça = gordura (heurística por cor)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    gord = ((s <= s_max) & (v >= v_min) & (peca > 0)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gord = cv2.morphologyEx(gord, cv2.MORPH_OPEN, k)
    # remove pontinhos (reflexo de luz costuma virar ruído)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gord, connectivity=8)
    limpa = np.zeros_like(gord)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            limpa[labels == i] = 1
    return limpa


def overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cor = np.zeros_like(img_bgr)
    for i, nome in enumerate(C.CLASSES):
        if i == 0:
            continue
        cor[mask == i] = C.COLORS.get(nome, (0, 255, 0))
    out = img_bgr.copy()
    fg = mask > 0
    out[fg] = cv2.addWeighted(img_bgr, 0.5, cor, 0.5, 0)[fg]
    for i in range(1, len(C.CLASSES)):
        cont, _ = cv2.findContours((mask == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="pasta com images/ e labels/ (ex.: data/train)")
    ap.add_argument("--model", default="sam_b.pt",
                    help="sam_b.pt (melhor) ou mobile_sam.pt (mais rápido)")
    ap.add_argument("--gordura", action="store_true",
                    help="separa a gordura por cor dentro da peça (valor 2)")
    ap.add_argument("--s-max", type=int, default=70, help="saturação máxima da gordura (0-255)")
    ap.add_argument("--v-min", type=int, default=150, help="brilho mínimo da gordura (0-255)")
    ap.add_argument("--preview", action="store_true", help="salva overlays em <split>/preview/")
    args = ap.parse_args()

    split = Path(args.split)
    labels_dir, masks_dir = split / "labels", split / "masks"
    prev_dir = split / "preview"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {args.model} ({device})...")
    sam = SAM(args.model)

    imagens = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    feitas = sem_txt = vazias = 0
    for img_path in imagens:
        txt = labels_dir / f"{img_path.stem}.txt"
        if not txt.exists():
            sem_txt += 1
            print(f"sem anotação: {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        caixas, poligonos = ler_caixas(txt, w, h)
        peca = np.zeros((h, w), np.uint8)

        if caixas:
            res = sam(str(img_path), bboxes=caixas, device=device, verbose=False)[0]
            if res.masks is not None:
                for m in res.masks.data.cpu().numpy():
                    m = (m > 0.5).astype(np.uint8)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    peca |= preencher_buracos(m)
        for poly in poligonos:
            cv2.fillPoly(peca, [poly], 1)

        if peca.sum() == 0:
            vazias += 1
            print(f"[AVISO] SAM não achou nada em {img_path.name}")

        mask = peca * CARNE
        if args.gordura:
            mask[detectar_gordura(img, peca, args.s_max, args.v_min) > 0] = GORDURA

        cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), mask)
        if args.preview:
            cv2.imwrite(str(prev_dir / f"{img_path.stem}.jpg"), overlay(img, mask))

        pct_carne = 100 * (mask == CARNE).mean()
        pct_gord = 100 * (mask == GORDURA).mean()
        print(f"[OK] {img_path.name}: carne {pct_carne:.1f}% | gordura {pct_gord:.1f}%")
        feitas += 1

    print(f"\n{feitas} máscaras criadas em {masks_dir}/")
    if args.preview:
        print(f"Confira as imagens em {prev_dir}/ antes de treinar.")
    if sem_txt:
        print(f"{sem_txt} imagens sem .txt foram puladas.")
    if vazias:
        print(f"{vazias} máscaras ficaram vazias — revise essas caixas.")


if __name__ == "__main__":
    main()
