"""Inferência + pós-processamento com OpenCV.
 
Uso:
    python predict.py --weights checkpoints/best.pt --source foto.jpg
    python predict.py --weights checkpoints/best.pt --source pasta_de_fotos/
    python predict.py --weights checkpoints/best.pt --source 0      # webcam
"""
import argparse
import json
from pathlib import Path
 
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
 
import config as C
from dataset import IMG_EXTS, preprocess
 
 
class MeatSegmenter:
    def __init__(self, weights: str, device: str | None = None):
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.size = ckpt["img_size"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = smp.create_model(ckpt["arch"], encoder_name=ckpt["encoder"],
                                      encoder_weights=None, classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
 
    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Retorna máscara (H, W) com o índice da classe por pixel."""
        h, w = image_bgr.shape[:2]
        x = preprocess(image_bgr, self.size).to(self.device)
        logits = self.model(x)
        # Redimensiona as probabilidades (não a máscara) para bordas mais suaves.
        logits = torch.nn.functional.interpolate(logits, size=(h, w),
                                                 mode="bilinear", align_corners=False)
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return clean_mask(mask, len(self.classes))
 
 
def clean_mask(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Abertura/fechamento morfológico e remoção de regiões pequenas."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.MORPH_KERNEL,) * 2)
    out = np.zeros_like(mask)
    for c in range(1, n_classes):
        m = (mask == c).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= C.MIN_AREA_PX:
                out[labels == i] = c
    return out
 
 
def measure(mask: np.ndarray, classes: list[str]) -> dict:
    """Área por classe e proporção de gordura (indicador de marmoreio)."""
    total = mask.size
    areas = {c: int((mask == i).sum()) for i, c in enumerate(classes)}
    result = {"area_px": areas,
              "percentual_imagem": {c: round(100 * a / total, 2) for c, a in areas.items()}}
    lean, fat = areas.get("carne_magra", 0), areas.get("gordura", 0)
    if lean + fat:
        result["gordura_sobre_carne_%"] = round(100 * fat / (lean + fat), 2)
    return result
 
 
def overlay(image_bgr: np.ndarray, mask: np.ndarray, classes: list[str],
            alpha: float = 0.45) -> np.ndarray:
    color = np.zeros_like(image_bgr)
    for i, c in enumerate(classes):
        if i == 0:
            continue
        color[mask == i] = C.COLORS.get(c, (0, 255, 0))
        contours, _ = cv2.findContours((mask == i).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color, contours, -1, (255, 255, 255), 1)
    blended = image_bgr.copy()
    fg = mask > 0
    blended[fg] = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)[fg]
    return blended
 
 
def draw_stats(img: np.ndarray, stats: dict) -> np.ndarray:
    y = 25
    for c, p in stats["percentual_imagem"].items():
        if c == "fundo":
            continue
        cv2.putText(img, f"{c}: {p:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25
    if "gordura_sobre_carne_%" in stats:
        cv2.putText(img, f"gordura/carne: {stats['gordura_sobre_carne_%']:.1f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img
 
 
def run_images(seg: MeatSegmenter, source: Path, out_dir: Path):
    files = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for p in files:
        img = cv2.imread(str(p))
        mask = seg.predict(img)
        stats = measure(mask, seg.classes)
        report[p.name] = stats
        cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{p.stem}_overlay.jpg"),
                    draw_stats(overlay(img, mask, seg.classes), stats))
        print(p.name, json.dumps(stats["percentual_imagem"], ensure_ascii=False))
    (out_dir / "relatorio.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Resultados em {out_dir}/")
 
 
def run_camera(seg: MeatSegmenter, index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {index}")
    print("Pressione 'q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = seg.predict(frame)
        vis = draw_stats(overlay(frame, mask, seg.classes), measure(mask, seg.classes))
        cv2.imshow("Segmentacao de carne", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/best.pt")
    ap.add_argument("--source", required=True, help="imagem, pasta ou índice da câmera")
    ap.add_argument("--out", default="resultados")
    args = ap.parse_args()
 
    seg = MeatSegmenter(args.weights)
    if args.source.isdigit():
        run_camera(seg, int(args.source))
    else:
        run_images(seg, Path(args.source), Path(args.out))
 
 
if __name__ == "__main__":
    main()