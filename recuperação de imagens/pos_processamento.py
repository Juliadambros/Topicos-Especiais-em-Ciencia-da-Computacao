from sklearn.cluster import KMeans

def obter_classe_nome(nome):
    partes = nome.split("_")
    return "_".join(partes[:-1])


def aplicar_pos_processamento_contexto(candidatos, nome_query, top_k=5, bonus_contexto=0.20):
    classe_query = obter_classe_nome(nome_query)

    candidatos_pos = []

    for cand in candidatos:
        classe_candidato = obter_classe_nome(cand["indice_original"])

        bonus = 0
        if classe_candidato == classe_query:
            bonus = bonus_contexto

        score_final = cand["similaridade_visual"] + bonus

        novo = cand.copy()
        novo["classe_query"] = classe_query
        novo["classe_candidato"] = classe_candidato
        novo["bonus_contexto"] = bonus
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