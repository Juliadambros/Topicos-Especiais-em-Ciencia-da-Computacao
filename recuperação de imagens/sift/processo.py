import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = 224

PASTA_BASE = "recuperação de imagens/sift/resultados"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PASTA_RESULTADOS = os.path.join(PASTA_BASE, f"execucao_{timestamp}")
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

#cria o objeto SIFT para extração de características locais
sift = cv2.SIFT_create()

def preprocessar_imagem(img_pil):
    img = img_pil.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)


def selecionar_regiao_query(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bbox = cv2.selectROI(
        "Selecione a regiao da query e pressione ENTER",
        img_bgr,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Selecione a regiao da query e pressione ENTER")

    x, y, w, h = bbox

    if w == 0 or h == 0:
        print("Nenhuma região selecionada. Usando centro como padrão.")
        return (56, 56, 112, 112)

    return (int(x), int(y), int(w), int(h))


def gerar_regioes_grid(img_size=IMG_SIZE, grid_size=3):
    regioes = []

    # Imagem inteira
    regioes.append((0, 0, img_size, img_size))

    largura_celula = img_size // grid_size
    altura_celula = img_size // grid_size

    for linha in range(grid_size):
        for coluna in range(grid_size):
            x = coluna * largura_celula
            y = linha * altura_celula
            regioes.append((x, y, largura_celula, altura_celula))

    return regioes


def extrair_descritores_sift(img, bbox):
    x, y, w, h = bbox
    #recorta região da imagem
    regiao = img[y:y+h, x:x+w]

    gray = cv2.cvtColor(regiao, cv2.COLOR_RGB2GRAY)

    #extrai keypoints e descritores
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return None

    return descriptors


def treinar_codebook(imagens_documentos, n_clusters=50):
    #junta todos descritores SIFT
    regioes = gerar_regioes_grid()
    todos_descritores = []

    for doc in imagens_documentos:
        img = doc["imagem"]

        for bbox in regioes:
            descriptors = extrair_descritores_sift(img, bbox)

            if descriptors is not None:
                todos_descritores.append(descriptors)

    if len(todos_descritores) == 0:
        raise ValueError("Nenhum descritor SIFT foi encontrado.")

    todos_descritores = np.vstack(todos_descritores)

    print(f"Total de descritores SIFT usados no K-means: {len(todos_descritores)}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    kmeans.fit(todos_descritores)

    return kmeans


def extrair_descritor_bovw(img, bbox, codebook):
    descriptors = extrair_descritores_sift(img, bbox)
    #cria histograma de palavras visuais (palavra visual é um grupo de características semelhantes extraídas das imagens)
    histograma = np.zeros(codebook.n_clusters)

    if descriptors is not None:
        palavras_visuais = codebook.predict(descriptors)

        for palavra in palavras_visuais:
            histograma[palavra] += 1

    histograma = histograma / (np.linalg.norm(histograma) + 1e-8)

    return histograma


def calcular_iou(box_a, box_b):
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

#armazena os descritores de todas as regiões de todas as imagens
def indexar_documentos(imagens_documentos, codebook):
    
    indice = []
    regioes = gerar_regioes_grid()

    for doc_id, doc in enumerate(imagens_documentos):
        img = doc["imagem"]
        label = doc["label"]
        indice_original = doc["indice_original"]

        for bbox in regioes:
            descritor = extrair_descritor_bovw(img, bbox, codebook)

            indice.append({
                "doc_id": doc_id,
                "indice_original": indice_original,
                "label": label,
                "bbox": bbox,
                "descritor": descritor,
                "imagem": img
            })

    return indice


def buscar_query(query_img, indice, codebook, top_k=5, bbox_query=None):
    if bbox_query is None:
        bbox_query = (56, 56, 112, 112)

    descritor_query = extrair_descritor_bovw(query_img, bbox_query, codebook)

    candidatos = []

    for item in indice:
        similaridade_visual = cosine_similarity(
            [descritor_query],
            [item["descritor"]]
        )[0][0]

        iou = calcular_iou(bbox_query, item["bbox"])

        candidatos.append({
            "doc_id": item["doc_id"],
            "indice_original": item["indice_original"],
            "label": item["label"],
            "imagem": item["imagem"],
            "bbox": item["bbox"],
            "similaridade_visual": similaridade_visual,
            "iou": iou
        })

    ranking_score = gerar_ranking_sem_repetir(
        candidatos,
        chave_ordenacao="similaridade_visual",
        top_k=top_k
    )

    ranking_iou = gerar_ranking_sem_repetir(
        candidatos,
        chave_ordenacao="iou",
        top_k=top_k
    )

    return ranking_score, ranking_iou


def gerar_ranking_sem_repetir(candidatos, chave_ordenacao, top_k=5):
    if chave_ordenacao == "iou":
        candidatos_ordenados = sorted(
            candidatos,
            key=lambda x: (x["iou"], x["similaridade_visual"]),
            reverse=True
        )
    else:
        candidatos_ordenados = sorted(
            candidatos,
            key=lambda x: x[chave_ordenacao],
            reverse=True
        )

    resultados = []
    documentos_usados = set()

    for cand in candidatos_ordenados:
        if cand["doc_id"] not in documentos_usados:
            resultados.append(cand)
            documentos_usados.add(cand["doc_id"])

        if len(resultados) == top_k:
            break

    return resultados


def desenhar_caixa(img, bbox):
    img_copy = img.copy()
    x, y, w, h = bbox

    cv2.rectangle(
        img_copy,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    return img_copy


def salvar_resultados(
    query_img,
    resultados,
    nome_query,
    indice_query=None,
    label_query=None,
    bbox_query=None,
    titulo_ranking="Ranking"
):
    if bbox_query is None:
        bbox_query = (56, 56, 112, 112)

    plt.figure(figsize=(16, 4))

    query_com_caixa = desenhar_caixa(query_img, bbox_query)

    plt.subplot(1, 6, 1)
    plt.imshow(query_com_caixa)

    titulo = f"{titulo_ranking}\nQuery\nIdx: {indice_query}"

    if label_query is not None:
        titulo += f"\nClasse: {label_query}"

    plt.title(titulo, fontsize=9)
    plt.axis("off")

    for i, res in enumerate(resultados):
        if "IoU" in titulo_ranking:
            img = desenhar_caixa(res["imagem"], res["bbox"])
        else:
            img = res["imagem"]

        plt.subplot(1, 6, i + 2)
        plt.imshow(img)

        if "IoU" in titulo_ranking:
            texto = (
                f"Top {i+1}\n"
                f"Idx: {res['indice_original']}\n"
                f"IoU: {res['iou']:.2f}"
            )
        else:
            texto = (
                f"Top {i+1}\n"
                f"Idx: {res['indice_original']}\n"
                f"Sim: {res['similaridade_visual']:.2f}"
            )

        plt.title(texto, fontsize=9)
        plt.axis("off")

    caminho = os.path.join(
        PASTA_RESULTADOS,
        f"resultado_{nome_query}.png"
    )

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()

    print(f"Imagem salva em: {caminho}")