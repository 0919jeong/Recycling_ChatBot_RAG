import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time

# --- 1. 문서 로드 ---
# 'documents' 폴더 내의 모든 .md 파일을 로드합니다.
print("1. 'documents' 폴더에서 .md 파일 로드를 시작합니다...")
loader = DirectoryLoader(
    './documents', 
    glob="**/*.md", 
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True,
    use_multithreading=True
)

try:
    docs = loader.load()
    if not docs:
        print("[오류] 'documents' 폴더에 파일이 없거나 .md 파일을 읽을 수 없습니다.")
        print("폴더와 파일이 올바르게 위치해 있는지 확인하세요.")
        exit()
    print(f"  > 총 {len(docs)}개의 문서를 성공적으로 로드했습니다.")

except Exception as e:
    print(f"[오류] 문서 로드 중 예외가 발생했습니다: {e}")
    print("unstructured, unstructured-markdown 라이브러리가 올바르게 설치되었는지 확인하세요.")
    exit()


# --- 2. 청킹 (Chunking) 생략 ---
# 분석 결과, 문서 크기가 작고 주제별로 나뉘어 있으므로 청킹을 생략합니다.
# 문서 1개 = 청크 1개
print("2. 문서 크기 분석 결과, 청킹(분할)을 생략합니다. (문서 1개 = 1 청크)")
split_docs = docs


# --- 3. 임베딩 및 벡터 스토어 생성 ---
# OpenAI 임베딩 모델을 사용하여 문서를 벡터로 변환합니다.
# (이 과정에서 OpenAI API가 호출됩니다.)
print("3. OpenAI 임베딩 모델로 문서를 벡터화합니다. (API 호출 발생)")
start_time = time.time()
try:
    embeddings = OpenAIEmbeddings()
    
    # FAISS 벡터 스토어를 생성합니다.
    vector_store = FAISS.from_documents(split_docs, embeddings)

except Exception as e:
    print(f"[오류] 임베딩 또는 FAISS 생성 중 오류가 발생했습니다: {e}")
    print("OPENAI_API_KEY가 환경 변수에 올바르게 설정되었는지 확인하세요.")
    exit()

end_time = time.time()
print(f"  > 벡터화 완료. (소요 시간: {end_time - start_time:.2f}초)")


# --- 4. 로컬에 저장 ---
# 생성된 벡터 스토어를 파일로 저장합니다.
print("4. 벡터 스토어를 'my_faiss_index' 폴더에 저장합니다.")
vector_store.save_local("my_faiss_index")

print("\n[성공] 'my_faiss_index' 생성이 완료되었습니다.")
print("이제 'step_2_rag_api.py' 파일을 실행하여 RAG API 서버를 시작할 수 있습니다.")