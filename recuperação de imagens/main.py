from baixar_dataset import carregar_dataset
from processo import (
    preprocessar_imagem,
    selecionar_regiao_query,
    indexar_documentos,
    buscar_query,
    salvar_resultados,
    PASTA_RESULTADOS
)

INDICES_DOCUMENTOS = [
    0, 5, 10, 15, 20,
    30, 35, 40, 45, 50,
    60, 70, 80, 90, 100,
    120, 140, 160, 180, 200,
    220, 240, 260, 280, 999
]

INDICES_QUERIES = [25, 55, 105, 205, 1000]


def main():
    print("Carregando dataset...")
    dataset = carregar_dataset()

    print(f"Resultados serão salvos em: {PASTA_RESULTADOS}")

    print("Preparando documentos...")

    imagens_documentos = []

    #Pré-processa imagens do banco
    for indice_doc in INDICES_DOCUMENTOS:
        img_pil, label = dataset[indice_doc]
        img = preprocessar_imagem(img_pil)

        imagens_documentos.append({
            "imagem": img,
            "label": label,
            "indice_original": indice_doc
        })

    print(f"Total de documentos: {len(imagens_documentos)}")

    print("Indexando documentos...")
    #Cria índice com regiões candidatas
    indice = indexar_documentos(imagens_documentos)

    print(f"Total de regiões indexadas: {len(indice)}")
    
    #Executa consultas
    for indice_query in INDICES_QUERIES:
        img_pil, label_query = dataset[indice_query]
        query_img = preprocessar_imagem(img_pil)

        print(f"\nBuscando query {indice_query}")

        bbox_query = selecionar_regiao_query(query_img)

        ranking_score, ranking_iou = buscar_query(
            query_img,
            indice,
            top_k=5,
            bbox_query=bbox_query
        )

        salvar_resultados(
            query_img,
            ranking_score,
            nome_query=f"score_query_{indice_query}",
            indice_query=indice_query,
            label_query=label_query,
            bbox_query=bbox_query,
            titulo_ranking="Ranking por Score"
        )

        salvar_resultados(
            query_img,
            ranking_iou,
            nome_query=f"iou_query_{indice_query}",
            indice_query=indice_query,
            label_query=label_query,
            bbox_query=bbox_query,
            titulo_ranking="Ranking por IoU"
        )

    print("\nFinalizado!")
    print(f"Imagens salvas em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()