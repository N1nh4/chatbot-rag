# Importações básicas
import os

# loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Divisão de texto em blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Banco vetorial
from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import ChatOpenAI

# Cadeia RAG
from langchain.chains import RetrievalQA


# Caminho do PDF com regras oficiais do futebol.
CAMINHO_PDF = "guia-alana-abreu.pdf" 

# Carrega o PDF
loader = PyPDFLoader(CAMINHO_PDF)
documents = loader.load()

# Quantidade de páginas carregadas
print(len(documents))

# Divide os documentos em chunks menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))

# Inicializa embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=" "
)

# Cria o banco vetorial
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_regras_futebol"
)

# Cria o retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Inicializa o modelo de linguagem
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=" "
)

# Cria a cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Pergunta de teste
pergunta = "Um jogador pode usar a mão para marcar um gol"

# Executa a pergunta no agente RAG
resposta = qa_chain(pergunta)

print("Pergunta:")
print(pergunta)

print("\nResposta do Agente:")
print(resposta["result"])

print("\nTrechos utilizados como contexto:\n")
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Fonte: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}\n")
    print("Conteúdo:")
    print(doc.page_content)
    print("\n")