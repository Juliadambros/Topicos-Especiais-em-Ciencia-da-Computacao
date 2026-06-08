from sklearn.cluster import KMeans


def aplicar_pos_processamento_cluster(
    candidatos,
    descritor_query,
    top_k=5,
    n_clusters=3,
    bonus_cluster=0.15
):
    descritores = [c["descritor"] for c in candidatos]

    if len(descritores) < n_clusters:
        candidatos_ordenados = sorted(
            candidatos,
            key=lambda x: x["similaridade_visual"],
            reverse=True
        )
        return remover_repetidos_por_documento(candidatos_ordenados, top_k)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(descritores)
    cluster_query = kmeans.predict([descritor_query])[0]

    candidatos_pos = []

    for cand, cluster in zip(candidatos, clusters):
        bonus = 0

        if cluster == cluster_query:
            bonus = bonus_cluster

        score_final = cand["similaridade_visual"] + bonus

        novo = cand.copy()
        novo["cluster"] = int(cluster)
        novo["cluster_query"] = int(cluster_query)
        novo["bonus_cluster"] = bonus
        novo["score_final"] = score_final

        candidatos_pos.append(novo)

    candidatos_ordenados = sorted(
        candidatos_pos,
        key=lambda x: x["score_final"],
        reverse=True
    )

    return remover_repetidos_por_documento(candidatos_ordenados, top_k)


def remover_repetidos_por_documento(candidatos_ordenados, top_k=5):
    resultados = []
    documentos_usados = set()

    for cand in candidatos_ordenados:
        if cand["doc_id"] not in documentos_usados:
            resultados.append(cand)
            documentos_usados.add(cand["doc_id"])

        if len(resultados) == top_k:
            break

    return resultados