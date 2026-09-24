"""
----- Treino de U-Net / DeepLabV3+ com segmentation_models_pytorch------

Visualizador do modelo de segmentação: escolha uma imagem e veja o resultado.

Mostra lado a lado:
    Original | Resultado do modelo | Anotação (se existir máscara em ../masks/)
e as porcentagens de carne, gordura e osso.

Uso (dentro da pasta Segmentacao):
    python visualizar.py
    python visualizar.py --weights checkpoints/best.pt
    python visualizar.py --image data/val/images/foto.jpg

Opção "Grade": desenha em azul o contorno da carne e a divisão em regiões
(usa o grade_carne.py, que precisa estar na mesma pasta).
"""
import argparse
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import seg_config as C
from predict import MeatSegmenter, measure, overlay
from grade_carne import desenhar as desenhar_grade

PASTA_INICIAL = Path("data/val/images")


def bgr_para_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mascara_colorida(mask, classes):
    out = np.zeros((*mask.shape, 3), np.uint8)
    for i, c in enumerate(classes):
        if i:
            out[mask == i] = C.COLORS.get(c, (0, 255, 0))
    return out


def iou_por_classe(pred, gt, n):
    res = {}
    for i in range(1, n):
        inter = np.logical_and(pred == i, gt == i).sum()
        uniao = np.logical_or(pred == i, gt == i).sum()
        if uniao:
            res[i] = inter / uniao
    return res


class Visualizador:
    def __init__(self, root, weights):
        self.root = root
        self.root.title("Segmentação de Carnes — Visualizador")
        self.root.geometry("1400x800")
        self.root.config(bg="#1e1e1e")

        self.weights = weights
        self.seg = None
        self.paineis = []          # imagens RGB atuais (para redimensionar)
        self.tk_imgs = []
        self.resultado = None      # (nome, overlay_bgr, mask) para salvar
        self.modo_var = tk.StringVar(value="overlay")

        # Grade
        self.grade_var = tk.BooleanVar(value=False)
        self.grade_modo_var = tk.StringVar(value="regioes")
        self.grade_qtd_var = tk.IntVar(value=10)      # nº de regiões
        self.grade_passo_var = tk.IntVar(value=100)   # tamanho da célula (px)
        self.grade_numerar_var = tk.BooleanVar(value=False)
        self.grade_veu_var = tk.BooleanVar(value=False)

        self._ultimo_tamanho = None
        self._redraw_job = None
        self._montar_ui()
        # Redesenha só quando a área de imagens muda de tamanho (com atraso, para não travar)
        self.area.bind("<Configure>", self._on_resize)
        self._carregar_modelo()

    # ---------------- UI ----------------
    def _montar_ui(self):
        barra = tk.Frame(self.root, bg="#2b2b2b", padx=10, pady=8)
        barra.pack(fill="x")

        def btn(txt, cmd, cor):
            b = tk.Button(barra, text=txt, command=cmd, bg=cor, fg="white",
                          font=("Arial", 10, "bold"), padx=10, cursor="hand2")
            b.pack(side=tk.LEFT, padx=4)
            return b

        self.btn_abrir = btn("📂 Abrir Imagem", self.abrir_imagem, "#008CBA")
        btn("⬅ Anterior", lambda: self.navegar(-1), "#455a64")
        btn("Próxima ➡", lambda: self.navegar(1), "#455a64")
        btn("💾 Salvar Resultado", self.salvar, "#4CAF50")
        btn("🧠 Trocar Modelo", self.trocar_modelo, "#6a1b9a")

        tk.Label(barra, text="  Exibir:", bg="#2b2b2b", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        for txt, val in (("Sobreposição", "overlay"), ("Só máscara", "mask")):
            tk.Radiobutton(barra, text=txt, variable=self.modo_var, value=val, command=self._atualizar_modo,
                           bg="#2b2b2b", fg="white", selectcolor="#444", font=("Arial", 9)).pack(side=tk.LEFT)

        self.lbl_status = tk.Label(barra, text="", bg="#2b2b2b", fg="#00ff00", font=("Consolas", 9))
        self.lbl_status.pack(side=tk.RIGHT)

        # --- Barra da grade ---
        barra_g = tk.Frame(self.root, bg="#263238", padx=10, pady=6)
        barra_g.pack(fill="x")
        estilo = dict(bg="#263238", fg="white", selectcolor="#37474f", font=("Arial", 9),
                      activebackground="#263238", activeforeground="white")

        tk.Checkbutton(barra_g, text="🔷 Mostrar grade", variable=self.grade_var,
                       command=self._atualizar_modo, **{**estilo, "font": ("Arial", 10, "bold")}).pack(side=tk.LEFT)
        tk.Label(barra_g, text="   Tipo:", bg="#263238", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        for txt, val in (("Regiões", "regioes"), ("Quadriculado", "grade")):
            tk.Radiobutton(barra_g, text=txt, variable=self.grade_modo_var, value=val,
                           command=self._atualizar_modo, **estilo).pack(side=tk.LEFT)

        tk.Label(barra_g, text="   Nº regiões:", bg="#263238", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        sp1 = tk.Spinbox(barra_g, from_=2, to=60, width=4, textvariable=self.grade_qtd_var,
                         command=self._atualizar_modo)
        sp1.pack(side=tk.LEFT)
        sp1.bind("<Return>", lambda e: self._atualizar_modo())
        tk.Label(barra_g, text="   Tam. célula (px):", bg="#263238", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        sp2 = tk.Spinbox(barra_g, from_=20, to=500, increment=10, width=5, textvariable=self.grade_passo_var,
                         command=self._atualizar_modo)
        sp2.pack(side=tk.LEFT)
        sp2.bind("<Return>", lambda e: self._atualizar_modo())

        tk.Checkbutton(barra_g, text="Numerar", variable=self.grade_numerar_var,
                       command=self._atualizar_modo, **estilo).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(barra_g, text="Véu azul", variable=self.grade_veu_var,
                       command=self._atualizar_modo, **estilo).pack(side=tk.LEFT)

        self.area = tk.Frame(self.root, bg="#1e1e1e")
        self.area.pack(fill="both", expand=True, padx=8, pady=8)

        rodape = tk.Frame(self.root, bg="#2b2b2b", padx=10, pady=8)
        rodape.pack(fill="x")
        self.lbl_stats = tk.Label(rodape, text="Abra uma imagem para começar.", bg="#2b2b2b", fg="white",
                                  font=("Consolas", 11), justify="left", anchor="w")
        self.lbl_stats.pack(side=tk.LEFT, fill="x", expand=True)

        legenda = tk.Frame(rodape, bg="#2b2b2b")
        legenda.pack(side=tk.RIGHT)
        for c in C.CLASSES[1:]:
            b, g, r = C.COLORS.get(c, (0, 255, 0))
            tk.Label(legenda, text="  ", bg=f"#{r:02x}{g:02x}{b:02x}").pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legenda, text=c, bg="#2b2b2b", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)

    def status(self, txt):
        self.lbl_status.config(text=txt)
        self.root.update_idletasks()

    # ---------------- Modelo ----------------
    def _carregar_modelo(self):
        if not Path(self.weights).exists():
            self.status("modelo não encontrado")
            messagebox.showwarning("Modelo", f"Não achei {self.weights}.\nTreine com train.py ou use '🧠 Trocar Modelo'.")
            return
        self.status("carregando modelo...")
        try:
            self.seg = MeatSegmenter(self.weights)
            self.status(f"modelo: {self.weights} ({self.seg.device})")
        except Exception as e:
            self.seg = None
            self.status("erro ao carregar modelo")
            messagebox.showerror("Erro", f"Falha ao carregar o modelo:\n{e}")

    def trocar_modelo(self):
        f = filedialog.askopenfilename(title="Escolha o modelo (.pt)", initialdir="checkpoints",
                                       filetypes=[("PyTorch", "*.pt"), ("Todos", "*.*")])
        if f:
            self.weights = f
            self._carregar_modelo()
            if self.resultado:
                self.processar(self.img_atual)

    # ---------------- Imagens ----------------
    def abrir_imagem(self):
        ini = PASTA_INICIAL if PASTA_INICIAL.exists() else Path(".")
        f = filedialog.askopenfilename(title="Escolha uma imagem", initialdir=str(ini),
                                       filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
        if f:
            self.processar(Path(f))

    def navegar(self, passo):
        if not getattr(self, "img_atual", None):
            return
        pasta = self.img_atual.parent
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if self.img_atual in arquivos:
            i = (arquivos.index(self.img_atual) + passo) % len(arquivos)
            self.processar(arquivos[i])

    def processar(self, caminho: Path):
        if self.seg is None:
            messagebox.showwarning("Modelo", "Nenhum modelo carregado.")
            return
        img = cv2.imread(str(caminho))
        if img is None:
            messagebox.showerror("Erro", f"Não consegui abrir:\n{caminho}")
            return
        self.img_atual = caminho
        self.btn_abrir.config(state=tk.DISABLED)
        self.status(f"processando {caminho.name}...")
        threading.Thread(target=self._inferir, args=(caminho, img), daemon=True).start()

    def _inferir(self, caminho, img):
        try:
            mask = self.seg.predict(img)
            erro = None
        except Exception as e:
            mask, erro = None, e
        self.root.after(0, lambda: self._mostrar(caminho, img, mask, erro))

    def _mostrar(self, caminho, img, mask, erro):
        self.btn_abrir.config(state=tk.NORMAL)
        if erro:
            self.status("erro")
            messagebox.showerror("Erro na inferência", str(erro))
            return

        classes = self.seg.classes
        stats = measure(mask, classes)
        ov = overlay(img, mask, classes)
        self.resultado = (caminho.stem, ov, mask)

        # Máscara de anotação (se a imagem estiver em .../images e existir .../masks/<nome>.png)
        gt_path = caminho.parent.parent / "masks" / f"{caminho.stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None

        self._img, self._mask, self._gt, self._classes = img, mask, gt, classes
        self._montar_paineis()

        # Estatísticas
        linhas = [f"{caminho.name}   ({img.shape[1]}x{img.shape[0]})"]
        partes = [f"{c}: {p:.1f}%" for c, p in stats["percentual_imagem"].items() if c != "fundo"]
        linhas.append("Área na imagem →  " + "   ".join(partes))
        if "gordura_sobre_carne_%" in stats:
            linhas.append(f"Gordura / (carne + gordura): {stats['gordura_sobre_carne_%']:.1f}%")
        if gt is not None:
            ious = iou_por_classe(mask, gt, len(classes))
            if ious:
                linhas.append("IoU vs anotação →  " + "   ".join(f"{classes[i]}: {v:.2f}" for i, v in ious.items()))
        self.lbl_stats.config(text="\n".join(linhas))
        self.status(f"ok — {caminho.name}")

    def _atualizar_modo(self):
        if self.resultado:
            self._montar_paineis()

    def _aplicar_grade(self, base_bgr, mask):
        """Desenha contorno + grade azul sobre base_bgr usando a máscara."""
        try:
            qtd = int(self.grade_qtd_var.get())
            passo = int(self.grade_passo_var.get())
        except (tk.TclError, ValueError):
            qtd, passo = 10, 100
        res, n = desenhar_grade(base_bgr, mask, modo=self.grade_modo_var.get(),
                                n_regioes=qtd, passo=max(10, passo),
                                numerar=self.grade_numerar_var.get(),
                                preencher=0.25 if self.grade_veu_var.get() else 0.0)
        return res, n

    def _montar_paineis(self):
        img, mask, gt, classes = self._img, self._mask, self._gt, self._classes
        so_mask = self.modo_var.get() == "mask"
        if self.grade_var.get():
            # Com grade: desenha sobre a foto original (ou sobre a máscara colorida)
            base = mascara_colorida(mask, classes) if so_mask else img
            pred_vis, n = self._aplicar_grade(base, mask)
            titulo = f"Resultado + grade ({n} regiões)" if n else "Resultado + grade (carne não encontrada)"
            self.resultado = (self.resultado[0], pred_vis, mask)
        else:
            pred_vis = mascara_colorida(mask, classes) if so_mask else overlay(img, mask, classes)
            titulo = "Resultado do modelo"
            self.resultado = (self.resultado[0], overlay(img, mask, classes), mask)
        self.paineis = [("Original", bgr_para_rgb(img)), (titulo, bgr_para_rgb(pred_vis))]
        if gt is not None:
            gt_vis = mascara_colorida(gt, classes) if so_mask else overlay(img, gt, classes)
            self.paineis.append(("Anotação (máscara de referência)", bgr_para_rgb(gt_vis)))
        self._desenhar()

    def _on_resize(self, event):
        tamanho = (event.width, event.height)
        if tamanho == self._ultimo_tamanho:
            return
        self._ultimo_tamanho = tamanho
        if self._redraw_job:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(150, self._desenhar)

    def _desenhar(self):
        if not self.paineis:
            return
        for w in self.area.winfo_children():
            w.destroy()
        self.tk_imgs = []
        larg = max(200, self.area.winfo_width() // len(self.paineis) - 12)
        alt = max(200, self.area.winfo_height() - 30)
        for titulo, rgb in self.paineis:
            col = tk.Frame(self.area, bg="#1e1e1e")
            col.pack(side=tk.LEFT, fill="both", expand=True, padx=4)
            tk.Label(col, text=titulo, bg="#1e1e1e", fg="white", font=("Arial", 10, "bold")).pack()
            pil = Image.fromarray(rgb)
            pil.thumbnail((larg, alt))
            tkimg = ImageTk.PhotoImage(pil)
            self.tk_imgs.append(tkimg)
            tk.Label(col, image=tkimg, bg="#1e1e1e").pack(expand=True)

    # ---------------- Salvar ----------------
    def salvar(self):
        if not self.resultado:
            messagebox.showinfo("Salvar", "Nada para salvar ainda.")
            return
        nome, ov, mask = self.resultado
        pasta = Path("resultados")
        pasta.mkdir(exist_ok=True)
        sufixo = "_grade.jpg" if self.grade_var.get() else "_overlay.jpg"
        cv2.imwrite(str(pasta / f"{nome}{sufixo}"), ov)
        cv2.imwrite(str(pasta / f"{nome}_mask.png"), mask)
        messagebox.showinfo("Salvo", f"Salvo em {pasta.resolve()}:\n{nome}{sufixo}\n{nome}_mask.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/best.pt")
    ap.add_argument("--image", help="abre esta imagem direto")
    args = ap.parse_args()

    root = tk.Tk()
    app = Visualizador(root, args.weights)
    if args.image:
        root.after(300, lambda: app.processar(Path(args.image)))
    root.mainloop()


if __name__ == "__main__":
    main()
