import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#tema: livros que viraram filme/serie)
docs = [
    {
        "id": "A",
        "titulo": "Portal Livros & Telas",
        "texto": "portal geral sobre livros e suas adaptações para cinema e séries com notícias e resenhas variadas",
        "links": ["B", "D", "F", "H", "K"]
    },
    {
        "id": "B",
        "titulo": "Resenha: O Hobbit (livro)",
        "texto": "resenha do livro o hobbit de jrr tolkien uma aventura de fantasia com a jornada inesperada de bilbo bolseiro",
        "links": ["C"]
    },
    {
        "id": "C",
        "titulo": "Resenha: O Hobbit (trilogia de filmes)",
        "texto": "resenha da trilogia de filmes o hobbit dirigida por peter jackson adaptação do livro de tolkien para o cinema",
        "links": ["B"]
    },
    {
        "id": "D",
        "titulo": "Resenha: Harry Potter e a Pedra Filosofal (livro)",
        "texto": "resenha do livro harry potter e a pedra filosofal escrito por jk rowling magia e o universo de hogwarts",
        "links": ["E"]
    },
    {
        "id": "E",
        "titulo": "Resenha: Harry Potter (filmes)",
        "texto": "resenha dos filmes de harry potter adaptação dos livros de jk rowling para o cinema com elenco e direção",
        "links": ["D"]
    },
    {
        "id": "F",
        "titulo": "Resenha: Duna (livro)",
        "texto": "resenha do livro duna escrito por frank herbert ficção científica no planeta deserto de arrakis",
        "links": ["G"]
    },
    {
        "id": "G",
        "titulo": "Resenha: Duna (filme 2021)",
        "texto": "resenha do filme duna de 2021 dirigido por denis villeneuve adaptação do livro de frank herbert para o cinema",
        "links": ["F"]
    },
    {
        "id": "H",
        "titulo": "Top 10 Adaptações de Livros",
        "texto": "lista top 10 melhores adaptações de livros para filmes confira agora a lista definitiva cliquem aqui adaptações adaptações imperdível",
        "links": ["B", "C", "D", "E", "F", "G", "I", "J", "K"]
    },
    {
        "id": "I",
        "titulo": "Blog de Curadoria Literária",
        "texto": "blog de curadoria literária com indicações das melhores adaptações de livros para o cinema análises profundas e comparação com as obras originais",
        "links": ["F", "D"]
    },
    {
        "id": "J",
        "titulo": "Loja Livros & Filmes",
        "texto": "loja virtual compre livros e filmes em blu-ray edições especiais com frete grátis e promoção por tempo limitado",
        "links": ["B", "D", "F", "H"]
    },
    {
        "id": "K",
        "titulo": "Notícias de Adaptações 2026",
        "texto": "notícias sobre as próximas adaptações de livros para filmes e séries em 2026 com anúncios de elenco e datas de estreia",
        "links": ["A", "H"]
    },
]

df_paginas = pd.DataFrame(docs)

print("\n1) paginas da mini-web")
print(df_paginas[["id", "titulo", "texto", "links"]].to_string(index=False))


#consultas usadas no trabalho (3 consultas, gabarito de relevancia)
consultas = [
    {
        "nome": "Duna: livro e filme",
        "consulta": "livro duna filme",
        "relevantes": {"F", "G"}
    },
    {
        "nome": "Harry Potter: livro e filme",
        "consulta": "harry potter livro filme",
        "relevantes": {"D", "E"}
    },
    {
        "nome": "Melhores adaptações de livros",
        "consulta": "melhores adaptações de livros para filme",
        "relevantes": {"I"}
    },
]

print("\n2) consultas usadas neste trabalho")
for c in consultas:
    print(f"- {c['nome']!r}: {c['consulta']!r} | relevantes = {sorted(c['relevantes'])}")


#indice invertido simples

indice = {}
for doc in docs:
    palavras = doc["texto"].lower().replace(",", " ").replace(".", " ").split()
    for palavra in palavras:
        indice.setdefault(palavra, set()).add(doc["id"])

print("\n3) exemplo de indice invertido")
palavras_exemplo = ["livro", "filme", "adaptação", "duna", "harry", "potter", "resenha"]
for palavra in palavras_exemplo:
    print(f"{palavra}: {sorted(indice.get(palavra, []))}")


#grafo - usado para PageRank, HITS e graus

G = nx.DiGraph()
for doc in docs:
    G.add_node(doc["id"], titulo=doc["titulo"])
    for destino in doc["links"]:
        G.add_edge(doc["id"], destino)

df_links = pd.DataFrame(list(G.edges()), columns=["origem", "destino"])

print("\n4) links da mini-web")
print(df_links.to_string(index=False))

# desenho do grafo
plt.figure(figsize=(10, 9))
ordem_circulo = ["H", "A", "K", "J", "I", "F", "G", "C", "B", "D", "E"]
pos = nx.circular_layout(G, scale=1.0)
pos = {nid: pos[nid] for nid in G.nodes()}

import math
n = len(ordem_circulo)
pos = {nid: (math.cos(2*math.pi*i/n + math.pi/2), math.sin(2*math.pi*i/n + math.pi/2))
       for i, nid in enumerate(ordem_circulo)}

cores = []
for node in G.nodes():
    if node == "H":
        cores.append("#e07a5f")  
    elif node == "I":
        cores.append("#81b29a")  
    else:
        cores.append("#bcd4e6")  

nx.draw_networkx_edges(
    G, pos, arrowstyle="-|>", arrowsize=16, connectionstyle="arc3,rad=0.15",
    edge_color="#9a9a9a", width=1.2, node_size=2200, min_source_margin=20, min_target_margin=20,
)
nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=cores, edgecolors="#333333", linewidths=1.4)
nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold")

legenda = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e07a5f", markersize=14,
               label="H = página \u201clista rasa\u201d (linka demais, penalizada)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#81b29a", markersize=14,
               label="I = conteúdo bom e pouco linkado (bonificada)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#bcd4e6", markersize=14,
               label="demais páginas"),
]
plt.legend(handles=legenda, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False, fontsize=10)

plt.title("Mini-Web: Livros que viraram filme/série — grafo de links", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("Trabalho final/imgs",
            dpi=180,
            bbox_inches="tight")
plt.close()
print("\n[grafo salvo em grafo_mini_web.png]")


#graus do grafo (links recebidos / links enviados)

df_graus = pd.DataFrame({
    "id": list(dict(G.in_degree()).keys()),
    "links_recebidos": list(dict(G.in_degree()).values()),
})
df_graus["links_enviados"] = df_graus["id"].map(dict(G.out_degree()))
df_graus = df_graus.merge(df_paginas[["id", "titulo"]], on="id")
df_graus = df_graus[["id", "titulo", "links_recebidos", "links_enviados"]]
df_graus = df_graus.sort_values("links_recebidos", ascending=False).reset_index(drop=True)

media_out_degree = df_graus["links_enviados"].mean()
limiar_out_degree = media_out_degree * 2          # acima disso = "lista rasa" / link farm
limiar_in_degree_baixo = 1                          # in-degree <= 1 = pouco linkada
limiar_score_texto_alto = 0.5                       # score normalizado >= 0.5 = responde bem

print("\n5) links recebidos e enviados por pagina")
print(df_graus.to_string(index=False))
print(f"\nmedia de links enviados na mini-web: {media_out_degree:.2f}")
print(f"limiar para 'lista rasa' (2x a media): {limiar_out_degree:.2f}")


#pagerank e hits - tambem independem da consulta 

pagerank = nx.pagerank(G, alpha=0.85)
df_pr = pd.DataFrame({"id": list(pagerank.keys()), "pagerank": list(pagerank.values())})
df_pr = df_pr.merge(df_paginas[["id", "titulo"]], on="id")
df_pr = df_pr[["id", "titulo", "pagerank"]].sort_values("pagerank", ascending=False).reset_index(drop=True)
df_pr["posicao_pagerank"] = df_pr.index + 1

print("\n6) ranking por pagerank")
print(df_pr.round(6).to_string(index=False))

hubs, authorities = nx.hits(G, max_iter=2000, normalized=True)
df_hits = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "hub": [hubs[doc["id"]] for doc in docs],
    "authority": [authorities[doc["id"]] for doc in docs],
})
for col in ["hub", "authority"]:
    df_hits[col] = df_hits[col].apply(lambda x: 0 if abs(x) < 1e-7 else x)

ranking_authorities = df_hits.sort_values("authority", ascending=False).reset_index(drop=True)
ranking_authorities["posicao_authority"] = ranking_authorities.index + 1
ranking_hubs = df_hits.sort_values("hub", ascending=False).reset_index(drop=True)
ranking_hubs["posicao_hub"] = ranking_hubs.index + 1

print("\n7) hits - ranking de authorities")
print(ranking_authorities[["posicao_authority", "id", "titulo", "authority"]].round(6).to_string(index=False))
print("\n8) hits - ranking de hubs")
print(ranking_hubs[["posicao_hub", "id", "titulo", "hub"]].round(6).to_string(index=False))

# flag global de "lista rasa" 
df_spam = df_graus[["id", "links_enviados"]].copy()
df_spam["lista_rasa"] = (df_spam["links_enviados"] > limiar_out_degree).astype(int)


#funcao que processa uma consulta inteira

textos = [doc["texto"] for doc in docs]
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(textos)


def normalizar_coluna(df, coluna):
    maior = df[coluna].max()
    if maior == 0:
        return 0
    return df[coluna] / maior


def precision_k(tabela, k, relevantes, coluna_id="id"):
    topk = tabela.head(k)[coluna_id].tolist()
    acertos = sum(p in relevantes for p in topk)
    return topk, acertos, acertos / k


def processar_consulta(nome, consulta, relevantes):
    #ranking textual (TF-IDF + cosseno)
    q = vectorizer.transform([consulta])
    scores_texto = cosine_similarity(q, X).flatten()

    ranking_texto = pd.DataFrame({
        "id": [doc["id"] for doc in docs],
        "titulo": [doc["titulo"] for doc in docs],
        "score_texto": scores_texto,
    }).sort_values("score_texto", ascending=False).reset_index(drop=True)
    ranking_texto["posicao_texto"] = ranking_texto.index + 1
    ranking_texto["relevante"] = ranking_texto["id"].isin(relevantes)

    #combinacao texto + pagerank + authority (normalizado + media ponderada)
    df_rank = ranking_texto[["id", "titulo", "score_texto"]].merge(df_pr[["id", "pagerank"]], on="id")
    df_rank = df_rank.merge(df_hits[["id", "hub", "authority"]], on="id")
    df_rank = df_rank.merge(df_graus[["id", "links_recebidos", "links_enviados"]], on="id")
    df_rank = df_rank.merge(df_spam[["id", "lista_rasa"]], on="id")

    for col in ["score_texto", "pagerank", "authority"]:
        df_rank[col + "_norm"] = normalizar_coluna(df_rank, col)

    peso_texto, peso_pagerank, peso_authority = 0.55, 0.25, 0.20
    df_rank["score_final"] = (
        peso_texto * df_rank["score_texto_norm"]
        + peso_pagerank * df_rank["pagerank_norm"]
        + peso_authority * df_rank["authority_norm"]
    )

    ranking_antes = df_rank.sort_values("score_final", ascending=False).reset_index(drop=True)
    ranking_antes["posicao_antes"] = ranking_antes.index + 1
    ranking_antes["relevante"] = ranking_antes["id"].isin(relevantes)

   
    # regra 1: penaliza "lista rasa" (links_enviados muito acima da media) -> perde 50% do score
    # regra 2: bonus para "conteudo esquecido" (links_recebidos baixo E score_texto_norm alto) -> ganha 30%
    df_rank["bonus_conteudo_esquecido"] = (
        (df_rank["links_recebidos"] <= limiar_in_degree_baixo)
        & (df_rank["score_texto_norm"] >= limiar_score_texto_alto)
    ).astype(int)

    fator = (1 - 0.50 * df_rank["lista_rasa"]) * (1 + 0.30 * df_rank["bonus_conteudo_esquecido"])
    df_rank["score_final_seguro"] = df_rank["score_final"] * fator

    ranking_depois = df_rank.sort_values("score_final_seguro", ascending=False).reset_index(drop=True)
    ranking_depois["posicao_depois"] = ranking_depois.index + 1
    ranking_depois["relevante"] = ranking_depois["id"].isin(relevantes)


    comparacao = ranking_antes[["id", "titulo", "posicao_antes"]].merge(
        ranking_depois[["id", "posicao_depois"]], on="id"
    )
    comparacao["mudanca"] = comparacao["posicao_antes"] - comparacao["posicao_depois"]
    comparacao = comparacao.merge(
        df_rank[["id", "lista_rasa", "bonus_conteudo_esquecido", "score_final", "score_final_seguro"]], on="id"
    )
    comparacao = comparacao.sort_values("posicao_depois").reset_index(drop=True)

    #precision@3
    top3_texto, acertos3_texto, p3_texto = precision_k(ranking_texto, 3, relevantes)
    top3_antes, acertos3_antes, p3_antes = precision_k(ranking_antes, 3, relevantes)
    top3_depois, acertos3_depois, p3_depois = precision_k(ranking_depois, 3, relevantes)

    ids_lista_rasa = set(df_spam.loc[df_spam["lista_rasa"] == 1, "id"])

    precisao = pd.DataFrame([
        {"ranking": "somente texto", "top3": ", ".join(top3_texto), "acertos": acertos3_texto, "precision@3": p3_texto,
         "lista_rasa_no_top3": any(p in ids_lista_rasa for p in top3_texto)},
        {"ranking": "texto + links (antes)", "top3": ", ".join(top3_antes), "acertos": acertos3_antes, "precision@3": p3_antes,
         "lista_rasa_no_top3": any(p in ids_lista_rasa for p in top3_antes)},
        {"ranking": "texto + links + melhoria (depois)", "top3": ", ".join(top3_depois), "acertos": acertos3_depois, "precision@3": p3_depois,
         "lista_rasa_no_top3": any(p in ids_lista_rasa for p in top3_depois)},
    ])

    print(f"\n================ CONSULTA: {nome} -> {consulta!r} ================")
    print(f"paginas relevantes (gabarito): {sorted(relevantes)}")

    print("\nranking textual:")
    print(ranking_texto[["posicao_texto", "id", "titulo", "score_texto", "relevante"]].round(6).to_string(index=False))

    print("\nranking final ANTES da melhoria (texto + pagerank + authority normalizados):")
    cols_antes = ["posicao_antes", "id", "titulo", "score_texto_norm", "pagerank_norm", "authority_norm", "score_final", "relevante"]
    print(ranking_antes[cols_antes].round(6).to_string(index=False))

    print("\nranking final DEPOIS da melhoria (lista rasa penalizada / conteudo esquecido bonificado):")
    cols_depois = ["posicao_depois", "id", "titulo", "score_final", "lista_rasa", "bonus_conteudo_esquecido", "score_final_seguro", "relevante"]
    print(ranking_depois[cols_depois].round(6).to_string(index=False))

    print("\ncomparacao antes/depois:")
    print(comparacao.round(6).to_string(index=False))

    print("\nprecision@3:")
    print(precisao.round(4).to_string(index=False))

    return {
        "nome": nome,
        "consulta": consulta,
        "relevantes": sorted(relevantes),
        "ranking_texto": ranking_texto[["posicao_texto", "id", "titulo", "score_texto", "relevante"]].round(6).to_dict("records"),
        "ranking_antes": ranking_antes[cols_antes].round(6).to_dict("records"),
        "ranking_depois": ranking_depois[cols_depois].round(6).to_dict("records"),
        "comparacao": comparacao.round(6).to_dict("records"),
        "precisao": precisao.round(4).to_dict("records"),
    }


resultados_consultas = []
for c in consultas:
    resultado = processar_consulta(c["nome"], c["consulta"], c["relevantes"])
    resultados_consultas.append(resultado)

print("\n\n================ RESUMO ================")
for r in resultados_consultas:
    p3_antes = next(p for p in r["precisao"] if p["ranking"] == "texto + links (antes)")["precision@3"]
    p3_depois = next(p for p in r["precisao"] if p["ranking"] == "texto + links + melhoria (depois)")["precision@3"]
    print(f"{r['nome']:35s} | precision@3 antes = {p3_antes:.4f} | precision@3 depois = {p3_depois:.4f}")

export = {
    "paginas": df_paginas[["id", "titulo", "texto", "links"]].to_dict("records"),
    "links": df_links.to_dict("records"),
    "graus": df_graus.round(4).to_dict("records"),
    "pagerank": df_pr.round(6).to_dict("records"),
    "hits_authority": ranking_authorities[["posicao_authority", "id", "titulo", "authority"]].round(6).to_dict("records"),
    "hits_hub": ranking_hubs[["posicao_hub", "id", "titulo", "hub"]].round(6).to_dict("records"),
    "parametros_melhoria": {
        "media_out_degree": round(media_out_degree, 4),
        "limiar_out_degree": round(limiar_out_degree, 4),
        "limiar_in_degree_baixo": limiar_in_degree_baixo,
        "limiar_score_texto_alto": limiar_score_texto_alto,
        "penalidade_lista_rasa": 0.50,
        "bonus_conteudo_esquecido": 0.30,
        "pesos": {"texto": 0.55, "pagerank": 0.25, "authority": 0.20},
    },
    "consultas": resultados_consultas,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print("\n[resultados exportados para resultados.json]")
