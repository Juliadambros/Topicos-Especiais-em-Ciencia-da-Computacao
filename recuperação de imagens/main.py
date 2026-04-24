from baixar_dataset import carregar_dataset
from processo import (
    preprocessar_imagem,
    indexar_documentos,
    buscar_query,
    salvar_resultados
)

NUM_DOCUMENTOS = 25

#quais imagens serão queries
INDICES_QUERIES = [25, 26, 27, 28, 29]

def main():
    print("Carregando dataset...")

    dataset = carregar_dataset()

    print("Preparando documentos...")

    imagens_documentos = []

    #as 25 primeiras imagens
    for i in range(NUM_DOCUMENTOS):
        img_pil, label = dataset[i]
        img = preprocessar_imagem(img_pil)
        imagens_documentos.append(img)

    print(f"Total de documentos: {len(imagens_documentos)}")

    print("Indexando documentos...")
    indice = indexar_documentos(imagens_documentos)

    print(f"Total de regiões indexadas: {len(indice)}")

    print("Executando queries escolhidas manualmente...")

    for indice_query in INDICES_QUERIES:
        img_pil, label = dataset[indice_query]
        query_img = preprocessar_imagem(img_pil)

        resultados = buscar_query(query_img, indice, top_k=5)

        salvar_resultados(
            query_img,
            resultados,
            nome_query=f"query_{indice_query}"
        )

    print("Processo finalizado!")


if __name__ == "__main__":
    main()