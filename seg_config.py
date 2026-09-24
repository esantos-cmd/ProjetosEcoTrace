"""Configuração central do projeto de segmentação de carnes."""

# Classes na ordem do índice usado nas máscaras (pixel = índice da classe).
CLASSES = ["fundo", "carne_magra", "gordura", "osso"]

# Cor (BGR, formato OpenCV) de cada classe: usada no overlay e na conversão
# de máscaras coloridas exportadas por ferramentas como CVAT / Label Studio.
COLORS = {
    "fundo": (0, 0, 0),
    "carne_magra": (0, 0, 200),     # vermelho
    "gordura": (200, 230, 255),     # creme
    "osso": (200, 200, 200),        # cinza claro
}

# Modelo
ARCH = "Unet"              # "Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN"...
ENCODER = "resnet34"       # "efficientnet-b3", "resnet50", "mobilenet_v2"...
ENCODER_WEIGHTS = "imagenet"

# Treino
IMG_SIZE = 512             # precisa ser múltiplo de 32
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 2

# Pós-processamento
MIN_AREA_PX = 200          # remove regiões menores que isso (ruído)
MORPH_KERNEL = 5
