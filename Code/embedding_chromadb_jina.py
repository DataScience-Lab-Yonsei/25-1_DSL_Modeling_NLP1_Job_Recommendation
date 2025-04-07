#!/usr/bin/env python
# coding: utf-8
'''
서버 실행시 아래 우선 실행 필수
pip install openpyxl einops chromadb
apt update && apt install zip -y

Hugging Face 모델 파일 동적 다운로드시 오류 -> 캐시 디렉토리 삭제
rm -rf /root/.cache/huggingface/modules/transformers_modules/jinaai

폴더 생성 후 폴더 압축 코드 
zip -r /root/chroma_db_jina.zip /root/chroma_db_jina
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

###############################
# 0) Jina 모델 래퍼 클래스
###############################
class JinaEmbeddingModel:
    """
    jinaai/jina-embeddings-v3 모델 래퍼
    """
    def __init__(self, model_name="jinaai/jina-embeddings-v3", device=None, task="text-matching", use_fp16=False):
        """
        model_name: 문자열, 사용할 Jina 모델 이름 (기본값: jinaai/jina-embeddings-v3)
        device: 'cuda' 또는 'cpu'. 자동 감지 후 설정 가능.
        task: Jina 모델에서 사용할 태스크 타입(기본값: "text-matching")
        use_fp16: True시 fp16 모드(단, GPU 환경에서만 권장)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.task = task
        
        print(f"[JinaEmbeddingModel] loading {model_name} (task={self.task}, fp16={use_fp16}, device={self.device}) ...")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True  # Jina 모델은 custom code를 사용하므로
        ).to(self.device)
        
        self.model.eval()
        
        # FP16 옵션
        if use_fp16 and "cuda" in self.device:
            self.model = self.model.half()

        print("[JinaEmbeddingModel] loaded.")

    def encode(self, texts, max_length=8192, batch_size=32):
        """
        texts: List[str] - 임베딩할 텍스트 목록
        returns dict: {"dense_vecs": np.ndarray(shape=(N, D))}
        """
        all_embeddings = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            # Jina 모델에 맞춰 encoding
            with torch.no_grad():
                embs = self.model.encode(
                    batch_texts,
                    task=self.task,
                    max_length=max_length
                )
            # embs가 torch.Tensor일 경우 numpy 변환
            if not isinstance(embs, np.ndarray):
                embs = embs.detach().cpu().numpy()
            all_embeddings.extend(embs)
        return {"dense_vecs": np.array(all_embeddings)}

    def stop_self_pool(self):
        # 필요 시 멀티프로세싱 풀 종료 등 처리
        pass


###############################
# 1) ChromaDB 연결
###############################
# db 폴더 저장하고 싶은 경로 (BGE와 구분하려면 별도 폴더 사용)
CHROMA_DB_DIR = "./chroma_db_jina"
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
# 2) Jina 모델 로드
###############################
# CPU 환경 기준: device="cpu", use_fp16=False
# GPU 환경이라면 device="cuda", use_fp16=True 등을 적절히 조정
model = JinaEmbeddingModel(
    model_name="jinaai/jina-embeddings-v3",
    device="cuda",
    task="text-matching",
    use_fp16=False
)

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

        # 예: "255120-공고제목-0"
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

        # Jina 임베딩
        emb_out = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH_SIZE,
            max_length=8000
        )
        batch_embeddings = emb_out["dense_vecs"]  # numpy array

        # ChromaDB upsert
        # documents에 실제 텍스트를 넣음
        collection.upsert(
            documents=batch_texts,
            embeddings=[emb.tolist() for emb in batch_embeddings],
            metadatas=batch_metas,
            ids=batch_ids
        )

        print(f"[{start_idx}~{end_idx-1}] batch upsert 완료 (총 {len(batch_texts)}건)")

    print("✅ 최종적으로 모든 문서를 ChromaDB에 upsert 완료!")

    # 모델 리소스 정리 (필요 시)
    model.stop_self_pool()
