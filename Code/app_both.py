import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoModel
from FlagEmbedding import BGEM3FlagModel

##############################################
# [Jina 모델 래퍼] jinaai/jina-embeddings-v3
##############################################
class JinaEmbeddingModel:
    """
    jinaai/jina-embeddings-v3 모델 래퍼
    """
    def __init__(self, model_name="jinaai/jina-embeddings-v3", device=None, task="text-matching", use_fp16=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.task = task
        print(f"[JinaEmbeddingModel] loading {model_name} (task={self.task}, fp16={use_fp16}, device={self.device}) ...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        # fp16 옵션이 필요하다면 여기서 적용 가능 (device==cuda일 때)

        print("[JinaEmbeddingModel] loaded.")

    def encode(self, texts, max_length=8192, batch_size=32):
        """
        texts: List[str]
        returns dict: {"dense_vecs": np.ndarray(shape=(N, D))}
        """
        all_embeddings = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]
            with torch.no_grad():
                embs = self.model.encode(
                    batch_texts,
                    task=self.task,
                    max_length=max_length
                )
            embs = embs if isinstance(embs, np.ndarray) else embs.detach().cpu().numpy()
            all_embeddings.extend(embs)
        return {"dense_vecs": np.array(all_embeddings)}

##############################################
# [코사인 유사도] 헬퍼 함수
##############################################
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

##############################################
# [ChromaDB] 컬렉션 가져오는 함수
##############################################
def get_chroma_collection(db_path: str, collection_name="job_postings_collection"):
    client_chroma = chromadb.PersistentClient(path=db_path)
    return client_chroma.get_collection(collection_name)

##############################################
# [BGE 모델 로드] (BGEM3FlagModel)
##############################################
def load_bge_model():
    bge_model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=False,  # CPU 환경이라면 False 권장
        device="cpu"
    )
    def embed_fn(text: str):
        if not text.strip():
            text = " "
        out = bge_model.encode(
            [text],
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return out["dense_vecs"][0]
    return embed_fn

##############################################
# [Jina 모델 로드]
##############################################
def load_jina_model():
    jina_model = JinaEmbeddingModel(
        model_name="jinaai/jina-embeddings-v3",
        use_fp16=False,
        task="text-matching"
    )
    def embed_fn(text: str):
        if not text.strip():
            text = " "
        out = jina_model.encode([text], max_length=1024)
        return out["dense_vecs"][0]
    return embed_fn

##############################################
# [소프트 필터 → Top5] 추출 함수
##############################################
def get_top5_postings(soft_filter: dict, collection, embed_func, df_all) -> pd.DataFrame:
    """
    soft_filter 예시:
    {
      "주요업무": {"가중치": 0.5, "조건": ["데이터 분석"]},
      "자격요건및우대사항": {"가중치": 0.3, "조건": ["SQL"]},
      "혜택및복지": {"가중치": 0.2, "조건": ["유연근무"]}
    }
    collection: chromadb Collection
    embed_func: (str) -> (np.array)
    df_all: 전체 채용공고 정보 DataFrame
    """
    # 1) ChromaDB에서 모든 문서(임베딩) 조회
    docs = collection.get(
        include=["embeddings", "metadatas"],
        limit=999999
    )
    if len(docs["ids"]) == 0:
        return pd.DataFrame([])

    # 2) 사용자 입력 필드별 텍스트 임베딩
    keyword_embeddings = {}
    for field_name, info in soft_filter.items():
        emb_list = []
        for kw in info["조건"]:
            emb_list.append(embed_func(kw))
        keyword_embeddings[field_name] = emb_list

    # 3) 문서별 raw 유사도 계산
    sim_raw = {}
    for i, doc_id in enumerate(docs["ids"]):
        emb = np.array(docs["embeddings"][i], dtype=np.float32)
        meta = docs["metadatas"][i]
        j_id = meta["공고id"]        # 공고 식별자
        doc_type = meta["type"]      # "주요업무", "자격요건및우대사항", "혜택및복지", ...

        if j_id not in sim_raw:
            sim_raw[j_id] = {}

        if doc_type in soft_filter:
            kw_embs = keyword_embeddings[doc_type]
            if not kw_embs:
                raw_sim = 0.0
            else:
                scores = []
                for kw_vec in kw_embs:
                    scores.append(cosine_similarity(emb, kw_vec))
                raw_sim = np.mean(scores)
            sim_raw[j_id][doc_type] = raw_sim

    # 4) 최종 점수 합산 (가중합)
    final_scores = {}
    for j_id in sim_raw.keys():
        score_sum = 0.0
        for field_name, info in soft_filter.items():
            raw_val = sim_raw[j_id].get(field_name, 0.0)
            weight = info["가중치"]
            score_sum += raw_val * weight
        final_scores[j_id] = score_sum

    # 5) 상위 5개 ID 뽑기
    sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:5]
    if not sorted_ids:
        return pd.DataFrame([])

    # 6) df_all에서 해당 ID 로우 추출 + 최종점수 컬럼
    top_df = df_all[df_all["공고id"].isin(sorted_ids)].copy()
    top_df["최종점수"] = top_df["공고id"].apply(lambda x: round(final_scores.get(str(x), 0.0), 4))

    # 점수 순으로 정렬
    top_df = top_df.sort_values("최종점수", ascending=False).reset_index(drop=True)
    return top_df

##############################################
# [사용자 입력(한 행) → 소프트필터 dict] 구성
##############################################
def build_soft_filter(row):
    """
    row: user_input.xlsx 한 행
         - 시행
         - 주요업무, 주요업무중요도
         - 자격요건및우대사항, 자격요건및우대사항중요도
         - 혜택및복지, 혜택및복지중요도
    """
    task_text    = str(row["주요업무"]).strip()
    task_imp     = float(row["주요업무중요도"]) if row["주요업무중요도"] else 0.0

    skill_text   = str(row["자격요건및우대사항"]).strip()
    skill_imp    = float(row["자격요건및우대사항중요도"]) if row["자격요건및우대사항중요도"] else 0.0

    benefit_text = str(row["혜택및복지"]).strip()
    benefit_imp  = float(row["혜택및복지중요도"]) if row["혜택및복지중요도"] else 0.0

    soft_list = []

    if task_text and task_imp > 0:
        soft_list.append(("주요업무", [task_text], task_imp))

    if skill_text and skill_imp > 0:
        soft_list.append(("자격요건및우대사항", [skill_text], skill_imp))

    if benefit_text and benefit_imp > 0:
        soft_list.append(("혜택및복지", [benefit_text], benefit_imp))

    # 가중치 합을 1로 비율화
    total_imp = sum(x[2] for x in soft_list)
    soft_filter_dict = {}
    if total_imp > 0:
        for field_name, kw_list, imp_val in soft_list:
            w = round(imp_val / total_imp, 4)
            soft_filter_dict[field_name] = {
                "가중치": w,
                "조건": kw_list
            }

    return soft_filter_dict

##############################################
# [전체 실행 함수]
##############################################
def process_user_inputs(embed_model_name: str,
                        df_user: pd.DataFrame,
                        df_all: pd.DataFrame):
    """
    embed_model_name: "bge" or "jina"
    1) 임베딩 모델 로드
    2) ChromaDB 컬렉션 가져오기
    3) df_user 각 행에 대해 소프트필터 → top5 추천
    4) (rank, 시행) 추가하여 누적 후 엑셀 저장
    """
    # 모델/DB 경로 결정
    if embed_model_name == "bge":
        embed_func = load_bge_model()
        db_path = "./chroma_db_bge"   # BGE용 ChromaDB
        output_filename = "bge_누적공고.xlsx"
    else:
        embed_func = load_jina_model()
        db_path = "./chroma_db_jina"  # Jina용 ChromaDB
        output_filename = "jina_누적공고.xlsx"

    # ChromaDB 컬렉션
    collection = get_chroma_collection(db_path, "job_postings_collection")

    all_outputs = []

    # user_input.xlsx 각 행 처리
    for idx, row in df_user.iterrows():
        run_id = row["시행"]            # "시행" 열

        # 소프트필터 생성
        soft_filter_dict = build_soft_filter(row)

        # 소프트필터가 비어 있으면 => Top5 추출 불가 => 빈 DF
        if not soft_filter_dict:
            tmp_df = df_all.head(0).copy()
            tmp_df["최종점수"] = 0.0
        else:
            # 상위 5개 추출
            tmp_df = get_top5_postings(soft_filter_dict, collection, embed_func, df_all)

        # rank, 시행 추가
        tmp_df.reset_index(drop=True, inplace=True)
        tmp_df["rank"] = tmp_df.index + 1
        tmp_df["시행"] = run_id

        # 컬럼 순서 맞추기 (원하는 순서대로)
        # df_all에 존재하는 컬럼 중 필요한 것만 고른 예시
        final_cols = [
            "시행", "rank", "공고id", "공고제목", "회사명",
            "주요업무", "자격요건", "우대사항", "혜택및복지",
            "근무위치", "경력", "최종점수"
        ]
        tmp_df = tmp_df[final_cols]
        all_outputs.append(tmp_df)

    if all_outputs:
        final_df = pd.concat(all_outputs, ignore_index=True)
    else:
        final_df = pd.DataFrame([])

    # 엑셀 저장
    final_df.to_excel(output_filename, index=False, engine="openpyxl")
    print(f"[{embed_model_name.upper()}] 최종 결과 → {output_filename}")

##############################################
# [main] 실행
##############################################
if __name__ == "__main__":
    # 1) user_input.xlsx 로드
    df_user = pd.read_excel("./user_input.xlsx")
    df_user = df_user.fillna("")

    # 2) 전체 공고 all_raw.xlsx 로드
    df_all = pd.read_excel("./all_raw.xlsx")
    df_all["공고id"] = df_all["공고id"].astype(str)

    # 3) BGE 모델로 추천
    process_user_inputs("bge", df_user, df_all)

    # 4) Jina 모델로 추천
    process_user_inputs("jina", df_user, df_all)

    print("\n모든 실행이 완료되었습니다.")
