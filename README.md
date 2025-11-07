# Recycling_ChatBot_RAG
분리수거 챗 봇의 RAG 소스코드 입니다.

## 1. 인덱싱
python indexing.py 로 1회만 실행합니다.
$env:OPENAI_API_KEY="sk-..." 를 통해 미리 환경변수에 API key 를 등록해주세요.

## 2. RAG 서버 실행
작업중입니다.

## 임시 작동 방법 정리
1. python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

$env:OPENAI_API_KEY="sk-..."

python indexing.py

uvicorn rag_api:app --reload