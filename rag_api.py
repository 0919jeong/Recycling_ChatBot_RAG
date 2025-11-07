# rag_api.py (LCEL v0.1+ 적용 버전)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os

# --- (변경 1) LCEL 구성을 위한 핵심 모듈 임포트 ---
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- 0. 환경 변수 확인 ---
if "OPENAI_API_KEY" not in os.environ:
    print("[오류] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("서버를 실행하기 전에 API 키를 설정해주세요.")
    exit()

# --- 1. FastAPI 앱 초기화 ---
app = FastAPI(
    title="My RAG API Server",
    description="24개 문서를 기반으로 질문에 답변하는 RAG API입니다.",
    version="1.0.0"
)

# --- 2. 모델 및 벡터 스토어 로드 (서버 시작 시 1회 실행) ---
try:
    print("RAG API 서버를 초기화합니다...")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "my_faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True 
    )
    
    # k=1: 가장 유사한 문서 1개를 가져옵니다.
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    print("  > LLM, 임베딩, 벡터스토어 로드 완료.")

except Exception as e:
    print(f"[치명적 오류] 모델 또는 벡터 스토어 로드에 실패했습니다: {e}")
    exit()


# --- (변경 2) 3. RAG 체인 생성 (LCEL 방식) ---

# 3-1. 프롬프트 템플릿 정의
# (기존 {input} 변수를 {question}으로 변경하여 LCEL 파이프라인과 호환되도록 함)
prompt_template = """
당신은 '분리배출 가이드' 전문 QA 챗봇입니다.
오직 제공되는 [참고 문서]의 내용을 기반으로만 질문에 답해야 합니다.
문서에 없는 내용은 절대 지어내지 말고, "제공된 문서에서는 해당 정보를 찾을 수 없습니다."라고 답하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# 3-2. LCEL 파이프라인 구성
# (이전의 create_retrieval_chain을 LCEL로 직접 구현)

# 1. 문서 검색 및 질문 통과 (입력: str -> 출력: {"context": docs, "question": str})
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# 2. 답변 생성 (입력: {"context": docs, "question": str} -> 출력: str)
#    ChatPromptTemplate가 context(docs 리스트)를 자동으로 문자열로 변환해 줌
answer_generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# 3. 최종 RAG 체인 결합
#    setup_and_retrieval의 출력을 받아,
#    "answer" 키에는 답변 생성 체인의 결과를,
#    "context" 키에는 원본 문서를 그대로 담아 반환
rag_chain = (
    setup_and_retrieval
    | RunnableParallel(
        answer=answer_generation_chain,
        context=(lambda x: x["context"])  # 원본 context(문서 리스트)를 그대로 통과
    )
)

print("  > RAG 체인 생성 완료 (LCEL 방식).")


# --- 4. API 엔드포인트 정의 (변경 없음) ---

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_document: str | None = None

@app.post("/ask", response_model=QueryResponse)
async def ask_rag(request: QueryRequest):
    """
    RAG 시스템에 질문을 하고 답변을 받습니다.
    """
    try:
        # LCEL 체인은 입력으로 질문(str)을 받음
        # (이전: {"input": request.query} -> 변경: request.query)
        response = await rag_chain.ainvoke(request.query)
        
        answer = response.get("answer", "답변을 생성하지 못했습니다.")
        
        # LCEL 체인의 출력 구조가 {"answer": str, "context": docs} 이므로
        # source 추출 로직은 그대로 작동함
        source = None
        if response.get("context"):
            source_path = response["context"][0].metadata.get("source")
            if source_path:
                source = os.path.basename(source_path)
                
        return QueryResponse(answer=answer, source_document=source)

    except Exception as e:
        return QueryResponse(answer=f"답변 생성 중 오류가 발생했습니다: {e}", source_document=None)

@app.get("/")
def read_root():
    return {"message": "RAG API 서버가 정상적으로 실행 중입니다. '/docs' 경로로 이동하여 API 문서를 확인하세요."}


# --- 5. 서버 실행 ---
if __name__ == "__main__":
    print("RAG API 서버를 http://127.0.0.1:8000 에서 시작합니다.")
    print("API 문서는 http://127.0.0.1:8000/docs 에서 확인할 수 있습니다.")
    uvicorn.run(app, host="127.0.0.1", port=8000)