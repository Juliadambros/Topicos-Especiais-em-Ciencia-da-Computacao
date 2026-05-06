import os
from PIL import Image

PASTA_IMAGENS = "recuperação de imagens/data/oxford-iiit-pet/images"


def carregar_dataset_por_nome():
    imagens = {}

    for arquivo in os.listdir(PASTA_IMAGENS):
        if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            caminho = os.path.join(PASTA_IMAGENS, arquivo)
            imagens[arquivo] = Image.open(caminho).convert("RGB")

    return imagens