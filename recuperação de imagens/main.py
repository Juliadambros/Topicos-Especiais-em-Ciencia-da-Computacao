from baixar_dataset import carregar_dataset_por_nome

from processo import (
    preprocessar_imagem,
    selecionar_regiao_query,
    indexar_documentos,
    buscar_query,
    salvar_resultados,
    PASTA_RESULTADOS
)

INDICES_DOCUMENTOS = [
    "Abyssinian_135", "Abyssinian_15", "Abyssinian_216", "Abyssinian_24",
    "Bengal_14", "Bengal_32", "Bengal_65", "Bengal_100",
    "Birman_4", "Birman_16", "Birman_41", "Birman_61",
    "Bombay_5", "Bombay_24", "Bombay_100", "Bombay_110",
    "British_Shorthair_42", "British_Shorthair_46",
    "Egyptian_Mau_40", "Egyptian_Mau_48", "Egyptian_Mau_73",
    "Siamese_151", "Siamese_158", "Siamese_108", "Siamese_139"
]

INDICES_QUERIES = [
    "Abyssinian_141",
    "Bengal_89",
    "Birman_7",
    "Bombay_140",
    "Siamese_65"
]


def main():

    print("Carregando imagens...")
    dataset = carregar_dataset_por_nome()

    print(f"Resultados serão salvos em: {PASTA_RESULTADOS}")

    print("Preparando documentos...")
    imagens_documentos = []

    for nome_doc in INDICES_DOCUMENTOS:

        arquivo = f"{nome_doc}.jpg"

        if arquivo not in dataset:
            print(f"Imagem não encontrada: {arquivo}")
            continue

        img_pil = dataset[arquivo]
        img = preprocessar_imagem(img_pil)

        imagens_documentos.append({
            "imagem": img,
            "label": nome_doc,
            "indice_original": nome_doc
        })

    print(f"Total de documentos: {len(imagens_documentos)}")

    print("Indexando documentos...")
    indice = indexar_documentos(imagens_documentos)

    print(f"Total de regiões indexadas: {len(indice)}")

    for nome_query in INDICES_QUERIES:

        arquivo_query = f"{nome_query}.jpg"

        if arquivo_query not in dataset:
            print(f"Query não encontrada: {arquivo_query}")
            continue

        img_pil = dataset[arquivo_query]
        query_img = preprocessar_imagem(img_pil)

        print(f"\nBuscando query: {nome_query}")

        bbox_query = selecionar_regiao_query(query_img)

        ranking_original, ranking_pos, ranking_iou = buscar_query(
            query_img,
            indice,
            top_k=5,
            bbox_query=bbox_query,
            nome_query=nome_query
        )

        salvar_resultados(
            query_img,
            ranking_original,
            nome_query=f"original_query_{nome_query}",
            indice_query=nome_query,
            label_query=nome_query,
            bbox_query=bbox_query,
            titulo_ranking="Ranking Original"
        )

        salvar_resultados(
            query_img,
            ranking_pos,
            nome_query=f"pos_query_{nome_query}",
            indice_query=nome_query,
            label_query=nome_query,
            bbox_query=bbox_query,
            titulo_ranking="Ranking Pos-Processado"
        )

        salvar_resultados(
            query_img,
            ranking_iou,
            nome_query=f"iou_query_{nome_query}",
            indice_query=nome_query,
            label_query=nome_query,
            bbox_query=bbox_query,
            titulo_ranking="Ranking por IoU"
        )

    print("\nFinalizado!")
    print(f"Imagens salvas em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()