from langchain_ollama import OllamaEmbeddings

texto = "Python é uma linguagem de programação."

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vetor_embedding = embeddings.embed_query(texto)

print("Texto:", texto)
print("Tamanho dos embeddings:", len(vetor_embedding))
print("Início do vetor (10 primeiros valores):", vetor_embedding[:10])
