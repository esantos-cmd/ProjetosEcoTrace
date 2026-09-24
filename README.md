# Segmentação de carnes: U-Net / DeepLabV3+
Uso (rode de dentro da pasta Segmentacao):

passo 1.
    python split_val.py --dry-run        # só mostra o que vai fazer
    python split_val.py                  # separa 30%
    python split_val.py --ratio 0.2      # separa 20%

passo 2.

cd ~/EcotraceProjeto/Projetos/Segmentacao
python yolo_to_masks.py --split data/train --gordura --preview
python yolo_to_masks.py --split data/val   --gordura --preview


passo 3.
python train.py --data data --epochs 50
python train.py --arch DeepLabV3Plus --encoder efficientnet-b3   # outra arquitetura


## Instalação
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# PyTorch com GPU NVIDIA (veja o comando certo em pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 1. Organizar os dados
```
data/
  train/images/001.jpg   train/masks/001.png
  val/images/101.jpg     val/masks/101.png
```
- A máscara tem o mesmo nome da imagem, em `.png`, com um valor por pixel:
  0 = fundo, 1 = carne_magra, 2 = gordura, 3 = osso (edite `CLASSES` em `config.py`).
- Separe cerca de 15–20% das imagens para `val`.
- Se a ferramenta de anotação exportar máscaras coloridas, ajuste `COLORS` em `config.py` e rode:
  ```bash
  python convert_masks.py --src export/SegmentationClass --dst data/train/masks
  ```

## 2. Treinar
```bash
python train.py --data data --epochs 50
python train.py --arch DeepLabV3Plus --encoder efficientnet-b3   # outra arquitetura
```
O melhor modelo (maior mIoU sem contar o fundo) é salvo em `checkpoints/best.pt`.

## 3. Inferir
```bash
python predict.py --source foto.jpg          # uma imagem
python predict.py --source pasta/            # pasta inteira
python predict.py --source 0                 # webcam ao vivo
```
Saídas em `resultados/`: máscara, overlay colorido e `relatorio.json` com a área de
cada classe e a **proporção gordura / (carne + gordura)**, um indicador de marmoreio.

## Dicas
- Iluminação e fundo padronizados valem mais que trocar de modelo.
- Com pouca GPU: reduza `--batch` ou use `--size 384`.
- Faixas finas de gordura (marmoreio): prefira `--size 768` ou mais, ou a arquitetura `UnetPlusPlus`.
