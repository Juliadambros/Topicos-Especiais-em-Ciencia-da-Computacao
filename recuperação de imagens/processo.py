import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = 224
PESO_VISUAL = 0.7
PESO_IOU = 0.3
PASTA_RESULTADOS = "recuperação de imagens/resultados"

os.makedirs(PASTA_RESULTADOS, exist_ok=True)


def preprocessar_imagem(img_pil):
    #redimensiona a imagem e converte para RGB
    img = img_pil.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)


def gerar_regioes_grid():
    #gera regiões candidatas fixas, cada região é uma caixa: x, y, largura, altura
    return [
        (0, 0, 224, 224), # imagem inteira
        (56, 56, 112, 112), # centro
        (0, 0, 112, 112), # superior esquerdo
        (112, 0, 112, 112), # superior direito
        (0, 112, 112, 112), # inferior esquerdo
        (112, 112, 112, 112), # inferior direito
    ]


def extrair_descritor_cor(img, bbox):
    x, y, w, h = bbox
    regiao = img[y:y+h, x:x+w]

    hist_r = cv2.calcHist([regiao], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([regiao], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([regiao], [2], None, [32], [0, 256])

    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    # Normalização
    hist = hist / (np.linalg.norm(hist) + 1e-8)

    return hist


def calcular_iou(box_a, box_b):
    #IoU = área de interseção / área de união.
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    area_intersecao = inter_w * inter_h
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    area_uniao = area_a + area_b - area_intersecao

    if area_uniao == 0:
        return 0

    return area_intersecao / area_uniao


def indexar_documentos(imagens_documentos):
    #Para cada imagem do banco, gera regiões candidatas, extrai descritores e armazena tudo em uma lista.
    indice = []
    regioes = gerar_regioes_grid()

    for doc_id, img in enumerate(imagens_documentos):
        for bbox in regioes:
            descritor = extrair_descritor_cor(img, bbox)

            indice.append({
                "doc_id": doc_id,
                "bbox": bbox,
                "descritor": descritor,
                "imagem": img
            })

    return indice


def buscar_query(query_img, indice, top_k=5, bbox_query=(56, 56, 112, 112)): # região central da query
    #Compara a query com todas as regiões indexadas. Score final: 70% similaridade visual 30% IoU
    descritor_query = extrair_descritor_cor(query_img, bbox_query)

    resultados = []

    for item in indice:
        similaridade_visual = cosine_similarity(
            [descritor_query],
            [item["descritor"]]
        )[0][0]

        iou = calcular_iou(bbox_query, item["bbox"])

        score_final = (PESO_VISUAL * similaridade_visual) + (PESO_IOU * iou)

        resultados.append({
            "doc_id": item["doc_id"],
            "imagem": item["imagem"],
            "bbox": item["bbox"],
            "similaridade_visual": similaridade_visual,
            "iou": iou,
            "score_final": score_final
        })

    resultados = sorted(resultados, key=lambda x: x["score_final"], reverse=True)
    return resultados[:top_k]


def desenhar_caixa(img, bbox):
    img_copy = img.copy()
    x, y, w, h = bbox

    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img_copy


def salvar_resultados(query_img, resultados, nome_query):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 6, 1)
    plt.imshow(query_img)
    plt.title(f"Query {nome_query}")
    plt.axis("off")

    for i, res in enumerate(resultados):
        img_com_caixa = desenhar_caixa(res["imagem"], res["bbox"])

        plt.subplot(1, 6, i + 2)
        plt.imshow(img_com_caixa)
        plt.title(
            f"Top {i+1}\n"
            f"Score: {res['score_final']:.2f}\n"
            f"IoU: {res['iou']:.2f}"
        )
        plt.axis("off")

    caminho = os.path.join(PASTA_RESULTADOS, f"resultado_{nome_query}.png")
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()

    print(f"Resultado salvo em: {caminho}")