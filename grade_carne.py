"""Desenha o contorno da carne e uma grade de regiões por dentro (estilo mapa de cortes).

A carne é localizada pelo modelo treinado (checkpoints/best.pt). Depois:
  1. o contorno externo da peça é desenhado em azul;
  2. o interior é dividido em regiões com linhas retas azuis, cortadas no contorno.

Modos da grade:
  regioes  -> polígonos irregulares (parecido com o mapa de cortes da carcaça)
  grade    -> linhas retas alinhadas ao eixo da peça (quadriculado)

Uso (dentro da pasta Segmentacao):
    python grade_carne.py --source data/val/images/foto.jpg
    python grade_carne.py --source data/val/images/            # pasta inteira
    python grade_carne.py --source foto.jpg --modo grade --passo 80
    python grade_carne.py --source foto.jpg --regioes 12 --numerar --mostrar
    python grade_carne.py --source foto.jpg --weights checkpoints_deeplab/best.pt
    python grade_carne.py --source data/train/images --usar-mascaras   # usa data/*/masks em vez do modelo
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import IMG_EXTS

AZUL = (255, 0, 0)  # BGR


# ------------------------------------------------------------------
# Máscara da carne
# ------------------------------------------------------------------
def maior_regiao(mask: np.ndarray) -> np.ndarray:
    """Fica só com a maior peça e preenche os buracos internos."""
    mask = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    if contornos:
        cv2.drawContours(out, [max(contornos, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return out


# ------------------------------------------------------------------
# Grades
# ------------------------------------------------------------------
def linhas_regioes(mask: np.ndarray, n_regioes: int, seed: int = 0):
    """Divide a peça em n regiões poligonais (Voronoi com sementes bem espalhadas).
    Retorna (camada_de_linhas, centros)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > 20000:  # subamostra para o k-means ficar rápido
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), 20000, replace=False)]

    n_regioes = max(2, min(n_regioes, len(pts) // 50))
    cv2.setRNGSeed(seed)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, _, centros = cv2.kmeans(pts, n_regioes, None, criterio, 3, cv2.KMEANS_PP_CENTERS)

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for cx, cy in centros:
        subdiv.insert((float(min(max(cx, 0), w - 1)), float(min(max(cy, 0), h - 1))))
    facetas, _ = subdiv.getVoronoiFacetList([])

    camada = np.zeros((h, w), np.uint8)
    for f in facetas:
        cv2.polylines(camada, [np.round(f).astype(np.int32)], True, 255, 1, cv2.LINE_AA)
    return camada, centros


def linhas_grade(mask: np.ndarray, passo: int):
    """Quadriculado alinhado ao eixo principal da peça. Retorna (camada, centros das células)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    media = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.vstack([xs - media[0], ys - media[1]]))
    _, vetores = np.linalg.eigh(cov)
    eixo_u = vetores[:, 1]              # direção do comprimento da peça
    eixo_v = np.array([-eixo_u[1], eixo_u[0]])

    diag = int(np.hypot(h, w))
    camada = np.zeros((h, w), np.uint8)
    centros = []
    for d in range(-diag, diag + 1, passo):
        for eixo, perp in ((eixo_u, eixo_v), (eixo_v, eixo_u)):
            p0 = media + perp * d - eixo * diag
            p1 = media + perp * d + eixo * diag
            cv2.line(camada, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 255, 1, cv2.LINE_AA)
    for a in range(-diag, diag + 1, passo):
        for b in range(-diag, diag + 1, passo):
            c = media + eixo_u * (a + passo / 2) + eixo_v * (b + passo / 2)
            x, y = int(c[0]), int(c[1])
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                centros.append((x, y))
    return camada, np.array(centros, np.float32)


# ------------------------------------------------------------------
# Desenho
# ------------------------------------------------------------------
def desenhar(img, mask, modo="regioes", n_regioes=10, passo=80, cor=AZUL,
             suavizar=0.004, numerar=False, preencher=0.0):
    h, w = img.shape[:2]
    esp = max(2, round(max(h, w) / 400))   # espessura proporcional à imagem
    out = img.copy()

    peca = maior_regiao(mask)
    if peca.sum() == 0:
        return out, 0

    # Contorno externo, levemente "facetado" como no mapa de cortes
    contornos, _ = cv2.findContours(peca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(contornos, key=cv2.contourArea)
    poligono = cv2.approxPolyDP(cont, suavizar * cv2.arcLength(cont, True), True)

    area_poligono = np.zeros_like(peca)
    cv2.fillPoly(area_poligono, [poligono], 1)

    if preencher > 0:  # leve véu azul dentro da peça
        veu = out.copy()
        veu[area_poligono > 0] = cor
        out = cv2.addWeighted(veu, preencher, out, 1 - preencher, 0)

    # Linhas internas, cortadas para ficarem só dentro da peça
    if modo == "grade":
        camada, centros = linhas_grade(area_poligono, passo)
    else:
        camada, centros = linhas_regioes(area_poligono, n_regioes)
    camada = cv2.dilate(camada, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (esp, esp)))
    dentro = cv2.erode(area_poligono, np.ones((3, 3), np.uint8))
    out[(camada > 0) & (dentro > 0)] = cor

    cv2.polylines(out, [poligono], True, cor, esp + 1, cv2.LINE_AA)

    if numerar:
        escala = max(0.5, max(h, w) / 1600)
        for i, (cx, cy) in enumerate(centros, 1):
            x, y = int(cx), int(cy)
            if 0 <= x < w and 0 <= y < h and area_poligono[y, x]:
                txt = str(i)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
                cv2.putText(out, txt, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            escala, (255, 255, 255), 2, cv2.LINE_AA)

    n = len(centros)
    return out, n


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="imagem ou pasta")
    ap.add_argument("--weights", default="checkpoints/best.pt")
    ap.add_argument("--usar-mascaras", action="store_true",
                    help="usa as máscaras de ../masks/ em vez do modelo")
    ap.add_argument("--modo", choices=["regioes", "grade"], default="regioes")
    ap.add_argument("--regioes", type=int, default=10, help="nº de regiões (modo regioes)")
    ap.add_argument("--passo", type=int, default=80, help="tamanho da célula em pixels (modo grade)")
    ap.add_argument("--simplificar", type=float, default=0.004,
                    help="simplificação do contorno (0 = contorno exato, 0.01 = mais reto)")
    ap.add_argument("--veu", type=float, default=0.0, help="preenchimento azul translúcido (0 a 1)")
    ap.add_argument("--numerar", action="store_true", help="escreve o número de cada região")
    ap.add_argument("--mostrar", action="store_true", help="abre uma janela com o resultado")
    ap.add_argument("--out", default="resultados_grade")
    args = ap.parse_args()

    src = Path(args.source)
    arquivos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not arquivos:
        print(f"Nenhuma imagem em {src}")
        return

    seg = None
    if not args.usar_mascaras:
        from predict import MeatSegmenter
        seg = MeatSegmenter(args.weights)
        print(f"Modelo: {args.weights}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in arquivos:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[erro] não abriu {p.name}")
            continue

        if seg is not None:
            mask = seg.predict(img)
        else:
            mp = p.parent.parent / "masks" / f"{p.stem}.png"
            if not mp.exists():
                print(f"[pulada] sem máscara: {mp}")
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        res, n = desenhar(img, mask, args.modo, args.regioes, args.passo,
                          suavizar=args.simplificar, numerar=args.numerar, preencher=args.veu)
        destino = out_dir / f"{p.stem}_grade.jpg"
        cv2.imwrite(str(destino), res)
        print(f"[OK] {p.name} -> {destino} ({n} regiões)" if n else f"[AVISO] carne não encontrada em {p.name}")

        if args.mostrar:
            vis = res.copy()
            esc = min(1.0, 1200 / max(vis.shape[:2]))
            vis = cv2.resize(vis, None, fx=esc, fy=esc)
            cv2.imshow("Grade da carne (qualquer tecla = proxima, q = sair)", vis)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    if args.mostrar:
        cv2.destroyAllWindows()
    print(f"\nResultados em {out_dir}/")


if __name__ == "__main__":
    main()
