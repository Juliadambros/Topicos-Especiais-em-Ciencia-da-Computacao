from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string

documento= [
    "Sistemas de inteligência artificial são utilizados para análise de dados e automação de tarefas.",
    "O aprendizado de máquina permite que computadores aprendam padrões a partir de dados.",
    "Redes neurais profundas são aplicadas em reconhecimento de imagem e processamento de linguagem natural.",
    "A segurança da informação é essencial para proteger dados em sistemas computacionais.",
    "Bancos de dados armazenam grandes volumes de informação e permitem consultas eficientes."
]

stopwords_basicas = {
    "o", "a", "e", "de", "do", "da", "para", "no", "na", "pelo", "em",
    "os", "as", "um", "uma", "que"
}

def limpar_texto(frase):
    frase = frase.lower()
    frase = frase.translate(str.maketrans('', '', string.punctuation))
    palavras = frase.split()
    palavras_filtradas = [p for p in palavras if p not in stopwords_basicas]
    return " ".join(palavras_filtradas)

documentos_processados = [limpar_texto(doc) for doc in documento]

print("documentos pré-processados")
for i, doc in enumerate(documentos_processados, start=1):
    print(f"Doc {i}: {doc}")
print()

#matriz termo-documento (TF bruto)
count_vectorizer = CountVectorizer()
matriz_tf = count_vectorizer.fit_transform(documentos_processados)
vocabulario = count_vectorizer.get_feature_names_out()

df_tf = pd.DataFrame(
    matriz_tf.toarray().T,
    index=vocabulario,
    columns=[f"Doc {i+1}" for i in range(len(documento))]
)

print("matriz termo-documento (TF bruto)")
print(df_tf)
print()

#matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer(norm='l2')
matriz_tfidf = tfidf_vectorizer.fit_transform(documentos_processados)
vocabulario_tfidf = tfidf_vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(
    matriz_tfidf.toarray().T,
    index=vocabulario_tfidf,
    columns=[f"Doc {i+1}" for i in range(len(documento))]
)

print("matriz TF-IDF")
print(df_tfidf.round(4))
print()

consulta = input("Digite sua consulta: ")
consulta_processada = limpar_texto(consulta)
print(consulta_processada)
print()

#vetor da consulta
vetor_consulta = tfidf_vectorizer.transform([consulta_processada])

df_consulta = pd.DataFrame(
    vetor_consulta.toarray(),
    columns=vocabulario_tfidf,
    index=["Consulta"]
)

print("vetor TF-IDF da consulta")
print(df_consulta.round(4))
print()

#similaridade de cosseno e ranking
similaridades = cosine_similarity(vetor_consulta, matriz_tfidf)[0]

ranking = sorted(
    [(i, score) for i, score in enumerate(similaridades)],
    key=lambda x: x[1],
    reverse=True
)

print("ranking")
for pos, score in ranking:
    print(f"Doc {pos+1} - Score: {score:.4f}")
    print(documento[pos])
    print()