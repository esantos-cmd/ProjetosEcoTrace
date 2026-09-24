"""Separa uma parte das imagens anotadas de data/train para data/val.

Move (não copia) cada imagem junto com o .txt de labels/ e a máscara de masks/
(se existir), para não ter a mesma imagem em treino e validação.

Antes de separar, as imagens de data/val que NÃO têm .txt são movidas para
data/sem_anotacao/images — o dataset.py exige máscara para toda imagem de val,
então imagens sem anotação ali quebram o treino.

Uso (rode de dentro da pasta Segmentacao):
    python split_val.py --dry-run        # só mostra o que vai fazer
    python split_val.py                  # separa 30%
    python split_val.py --ratio 0.2      # separa 20%
"""
import argparse
import random
import shutil
from pathlib import Path

from dataset import IMG_EXTS


def mover(src: Path, dst_dir: Path, dry: bool) -> bool:
    """Move src para dst_dir sem sobrescrever. Retorna True se moveu."""
    if not src.exists():
        return False
    dst = dst_dir / src.name
    if dst.exists():
        print(f"  [pulado] já existe em {dst_dir}: {src.name}")
        return False
    if not dry:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True


def imagens(pasta: Path):
    if not pasta.is_dir():
        return []
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="pasta com train/ e val/")
    ap.add_argument("--ratio", type=float, default=0.30, help="fração para validação (padrão 0.30)")
    ap.add_argument("--seed", type=int, default=42, help="semente para a separação ser reproduzível")
    ap.add_argument("--dry-run", action="store_true", help="só mostra, não move nada")
    args = ap.parse_args()

    data = Path(args.data)
    train, val = data / "train", data / "val"
    dry = args.dry_run
    if dry:
        print("=== MODO SIMULAÇÃO (nada será movido) ===\n")

    # 1) Tira de val as imagens sem anotação
    sem_anot_dir = data / "sem_anotacao" / "images"
    tirados = 0
    for img in imagens(val / "images"):
        if not (val / "labels" / f"{img.stem}.txt").exists():
            if mover(img, sem_anot_dir, dry):
                mover(val / "masks" / f"{img.stem}.png", data / "sem_anotacao" / "masks", dry)
                tirados += 1
    if tirados:
        print(f"{tirados} imagens sem anotação saíram de val/ -> {sem_anot_dir}/\n")

    # 2) Seleciona as imagens de train que têm .txt
    anotadas = [p for p in imagens(train / "images") if (train / "labels" / f"{p.stem}.txt").exists()]
    sem_txt = len(imagens(train / "images")) - len(anotadas)
    if not anotadas:
        print("Nenhuma imagem anotada em train/ (images/ + labels/). Nada a separar.")
        return

    n_val = max(1, round(len(anotadas) * args.ratio))
    if n_val >= len(anotadas):
        print(f"Poucas imagens ({len(anotadas)}) para separar {args.ratio:.0%}. Nada foi feito.")
        return

    random.seed(args.seed)
    escolhidas = random.sample(anotadas, n_val)

    # 3) Move imagem + label + máscara
    movidas = 0
    for img in escolhidas:
        if mover(img, val / "images", dry):
            mover(train / "labels" / f"{img.stem}.txt", val / "labels", dry)
            mover(train / "masks" / f"{img.stem}.png", val / "masks", dry)
            movidas += 1
            print(f"  -> val: {img.name}")

    print(f"\n{movidas} imagens movidas para val/ ({args.ratio:.0%} de {len(anotadas)} anotadas).")
    print(f"Ficaram em train/: {len(anotadas) - movidas} anotadas.")
    if sem_txt:
        print(f"Obs.: {sem_txt} imagens de train/ não têm .txt e não entraram na separação.")
    if dry:
        print("\n(simulação) Rode sem --dry-run para mover de verdade.")
    else:
        print("\nPróximo passo: gere as máscaras de novo:")
        print("    python yolo_to_masks.py --split data/train")
        print("    python yolo_to_masks.py --split data/val")


if __name__ == "__main__":
    main()
