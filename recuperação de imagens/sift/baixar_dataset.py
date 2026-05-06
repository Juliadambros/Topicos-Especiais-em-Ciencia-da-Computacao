import os
from torchvision.datasets import OxfordIIITPet

PASTA_DATASET = "recuperação de imagens/data"

def carregar_dataset():
    os.makedirs(PASTA_DATASET, exist_ok=True)

    dataset = OxfordIIITPet(
        root=PASTA_DATASET,
        split="trainval",
        target_types="category",
        download=True
    )

    return dataset