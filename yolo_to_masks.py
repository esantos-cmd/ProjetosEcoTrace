"""Converte anotações YOLO (.txt) em máscaras PNG para segmentação.

Aceita os dois formatos de linha (coordenadas normalizadas 0–1):
    Polígono:  <classe> x1 y1 x2 y2 x3 y3 ...   -> preenchido direto
    Caixa:     <classe> xc yc w h               -> o contorno da peça é recortado
                                                   dentro da caixa com GrabCut (OpenCV)

Valores na máscara (iguais ao seg_config.CLASSES):
    0 = fundo, 1 = carne_magra, 2 = gordura, 3 = osso

Caixas viram carne_magra (1). Com --gordura, as partes claras dentro da peça
viram gordura (2).

Estrutura:
    data/train/images/foto.jpg
    data/train/labels/foto.txt    <- suas anotações
    data/train/masks/foto.png     <- gerado por este script
    data/train/preview/foto.jpg   <- (com --preview) para conferir

Uso:
    python yolo_to_masks.py --split data/train --gordura --preview
    python yolo_to_masks.py --split data/val   --gordura --preview
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import IMG_EXTS
import seg_config as C

CARNE, GORDURA = 1, 2
LADO_GRABCUT = 800  # a imagem é reduzida para o GrabCut ficar rápido


def recortar_peca_grabcut(img, caixa, iteracoes=5):
    """Recorta o objeto dentro da caixa [x1, y1, x2, y2] (pixels) e devolve máscara 0/1."""
    h, w = img.shape[:2]
    esc = min(1.0, LADO_GRABCUT / max(h, w))
    peq = cv2.resize(img, (int(w * esc), int(h * esc))) if esc < 1 else img.copy()
    ph, pw = peq.shape[:2]

    x1, y1, x2, y2 = [int(round(v * esc)) for v in caixa]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pw - 1, x2), min(ph - 1, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return np.zeros((h, w), np.uint8)

    # Inicialização guiada por cor: fora da caixa = fundo; dentro da caixa,
    # pixels avermelhados/saturados = provável carne, o resto (inox, reflexo) = provável fundo.
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    vermelho = ((hue <= 15) | (hue >= 155)) & (sat >= 70)
    gc = np.full((ph, pw), cv2.GC_BGD, np.uint8)
    dentro = np.zeros((ph, pw), bool)
    dentro[y1:y2, x1:x2] = True
    gc[dentro] = cv2.GC_PR_BGD
    gc[dentro & vermelho] = cv2.GC_PR_FGD
    if not (gc == cv2.GC_PR_FGD).any():  # nada vermelho: volta ao modo retângulo
        gc = np.zeros((ph, pw), np.uint8)
        modo = cv2.GC_INIT_WITH_RECT
    else:
        modo = cv2.GC_INIT_WITH_MASK
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(peq, gc, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, iteracoes, modo)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    # Limpeza: fecha falhas, fica com a maior região e preenche buracos
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    if contornos:
        cv2.drawContours(m, [max(contornos, key=cv2.contourArea)], -1, 1, cv2.FILLED)

    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if esc < 1 else m


def detectar_gordura(img, peca, s_max, v_min):
    """Pixels claros e pouco saturados dentro da peça = gordura."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min) & (peca > 0)).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    out = np.zeros_like(g)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            out[lab == i] = 1
    return out


def preview(img, mask):
    cor = np.zeros_like(img)
    for i, nome in enumerate(C.CLASSES):
        if i:
            cor[mask == i] = C.COLORS.get(nome, (0, 255, 0))
    out = img.copy()
    fg = mask > 0
    out[fg] = cv2.addWeighted(img, 0.5, cor, 0.5, 0)[fg]
    for i in range(1, len(C.CLASSES)):
        cont, _ = cv2.findContours((mask == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="pasta com images/ e labels/")
    ap.add_argument("--gordura", action="store_true", help="separa gordura por cor dentro da peça")
    ap.add_argument("--s-max", type=int, default=70, help="saturação máxima da gordura (0-255)")
    ap.add_argument("--v-min", type=int, default=150, help="brilho mínimo da gordura (0-255)")
    ap.add_argument("--caixa-cheia", action="store_true",
                    help="não usa GrabCut: pinta o retângulo inteiro (menos preciso)")
    ap.add_argument("--preview", action="store_true", help="salva overlays em <split>/preview/")
    args = ap.parse_args()

    split = Path(args.split)
    labels_dir, masks_dir = split / "labels", split / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = split / "preview"
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(C.CLASSES)
    images = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    done = no_label = vazias = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        txt = labels_dir / f"{img_path.stem}.txt"
        if not txt.exists():
            no_label += 1
            print(f"sem anotação: {img_path.name}")
            continue

        peca = np.zeros((h, w), np.uint8)
        for line in txt.read_text().splitlines():
            vals = line.split()
            if len(vals) == 5:  # caixa
                xc, yc, bw, bh = map(float, vals[1:])
                caixa = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                if args.caixa_cheia:
                    x1, y1, x2, y2 = [int(round(v)) for v in caixa]
                    cv2.rectangle(peca, (x1, y1), (x2, y2), 1, cv2.FILLED)
                else:
                    peca |= recortar_peca_grabcut(img, caixa)
            elif len(vals) >= 7:  # polígono
                cls = int(vals[0]) + 1
                if cls >= n_classes:
                    cls = CARNE  # classe fora do seg_config vira carne
                pts = np.array(vals[1:], np.float32).reshape(-1, 2) * [w, h]
                cv2.fillPoly(mask, [pts.round().astype(np.int32)], cls)

        mask[peca > 0] = CARNE
        if args.gordura:
            mask[detectar_gordura(img, (mask > 0).astype(np.uint8), args.s_max, args.v_min) > 0] = GORDURA

        if mask.max() == 0:
            vazias += 1
            print(f"[AVISO] máscara vazia: {img_path.name}")

        cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), mask)
        if args.preview:
            cv2.imwrite(str(prev_dir / f"{img_path.stem}.jpg"), preview(img, mask))
        print(f"[OK] {img_path.name}: carne {100 * (mask == CARNE).mean():.1f}% | "
              f"gordura {100 * (mask == GORDURA).mean():.1f}%")
        done += 1

    print(f"\n{done} máscaras criadas em {masks_dir}/")
    if args.preview:
        print(f"Confira as imagens em {prev_dir}/ antes de treinar.")
    if no_label:
        print(f"{no_label} imagens sem .txt (anote ou remova de images/)")
    if vazias:
        print(f"AVISO: {vazias} máscaras ficaram vazias — revise essas anotações.")


if __name__ == "__main__":
    main()