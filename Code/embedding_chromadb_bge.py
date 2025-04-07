#!/usr/bin/env python
# coding: utf-8
'''
서버 실행시 아래 우선 실행 필수
pip install openpyxl FlagEmbedding chromadb
apt update && apt install zip -y

폴더 생성 후 폴더 압축 코드 
zip -r /root/chroma_db_bge.zip /root/chroma_db_bge
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from FlagEmbedding import BGEM3FlagModel

###############################
# 1) ChromaDB 연결
###############################
# db 폴더 저장하고 싶은 경로
CHROMA_DB_DIR = "./chroma_db_bge"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# db collection 이름 설정 
collection_name = "job_postings_collection"

try:
    # 이미 존재하면 get_collection
    collection = client.get_collection(collection_name)
except:
    # 없다면 create_collection
    collection = client.create_collection(collection_name)

###############################
# 2) bge-m3 모델 로드
###############################
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

###############################
# 3) CSV(or XLSX) 로드
###############################
df = pd.read_excel("./all.xlsx")

###############################
# 4) 임베딩 추출할 열 정의
###############################
COLUMNS_TO_EMBED = ["공고제목", "주요업무", "자격요건및우대사항", "혜택및복지"]

###############################
# 5) Batch 임베딩 준비
###############################
texts = []
metadatas = []
doc_ids = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="임베딩 준비 중", unit="row"):
    job_id = str(row["공고id"])
    exp_val = row.get("경력", 0)
    loc_val = row.get("근무위치", "")

    for col_type in COLUMNS_TO_EMBED:
        text_data = str(row.get(col_type, "")).strip()
        if not text_data:
            text_data = " "

        # doc_id 예: "255120-공고제목-0"
        doc_id = f"{job_id}-{col_type}-{i}"

        meta = {
            "공고id": job_id,
            "type": col_type,
            "경력": float(exp_val),
            "근무위치": loc_val
        }

        texts.append(text_data)
        doc_ids.append(doc_id)
        metadatas.append(meta)

print(f"✅ 총 {len(texts)}개 텍스트 임베딩 준비 완료.")

###############################
# 6) Batch 임베딩 & upsert
###############################
if __name__ == "__main__":
    BATCH_SIZE = 3000
    ENCODE_BATCH_SIZE = 32

    total_docs = len(texts)
    for start_idx in range(0, total_docs, BATCH_SIZE):
        end_idx = start_idx + BATCH_SIZE
        batch_texts = texts[start_idx:end_idx]
        batch_ids = doc_ids[start_idx:end_idx]
        batch_metas = metadatas[start_idx:end_idx]

        # bge-m3 임베딩
        emb_out = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH_SIZE,
            max_length=8000,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        batch_embeddings = emb_out["dense_vecs"]  # numpy array 형태

        # ChromaDB upsert
        # documents에 실제 텍스트를 넣도록 변경
        collection.upsert(
            documents=batch_texts,
            embeddings=[emb.tolist() for emb in batch_embeddings],
            metadatas=batch_metas,
            ids=batch_ids
        )

        print(f"[{start_idx}~{end_idx-1}] batch upsert 완료 (총 {len(batch_texts)}건)")

    print("✅ 최종적으로 모든 문서를 ChromaDB에 upsert 완료!")

    # 🔥 멀티프로세싱 풀 종료(FlagEmbedding 관련)
    model.stop_self_pool()
