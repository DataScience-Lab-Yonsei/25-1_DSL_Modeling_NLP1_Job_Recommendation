import streamlit as st
import openai
import json
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from FlagEmbedding import BGEM3FlagModel
import torch
from transformers import AutoModel

############################
# 0) GPT API Key
############################
openai.api_key = "개인_API_key"
client = openai.OpenAI(api_key=openai.api_key)

############################
# 0-1) [수정] jina 모델 래퍼
############################
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

    def stop_self_pool(self):
        pass  # 필요 시 멀티프로세싱 풀 종료 로직 등
        
# 들여쓰기 처리를 위한 기호 목록
INDENTATION_MARKERS = [
    "1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)", "-", "[",
    "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9.", "10. ",
    "■", "●", "ㆍ", "·", "•", "ㅇ", "“", "‘", "[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]",
    "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)", "○", "▪", "▶","•","【"
]

############################
# 1) 세션 상태 초기화
############################
if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "job_recommendation"

if "selected_sido" not in st.session_state:
    st.session_state["selected_sido"] = []

############################
# 2) 한국 시도/시군구 데이터
############################
location_dict = {
    "서울": ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
             "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구",
             "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구",
             "강남구", "송파구", "강동구"],
    "부산": ["중구", "서구", "동구", "영도구", "부산진구", "동래구", "남구", "북구",
             "해운대구", "사하구", "금정구", "강서구", "연제구", "수영구", "사상구",
             "기장군"],
    "대구": ["중구", "동구", "서구", "남구", "북구", "수성구", "달서구", "달성군", "군위군"],
    "인천": ["강화군", "옹진군", "중구", "동구", "미추홀구", "연수구", "남동구",
             "부평구", "계양구", "서구"],
    "광주": ["동구", "서구", "남구", "북구", "광산구"],
    "대전": ["동구", "중구", "서구", "유성구", "대덕구"],
    "울산": ["중구", "남구", "동구", "북구", "울주군"],
    "세종": [],
    "경기": ["수원시", "고양시", "용인시", "성남시", "부천시", "화성시", "안산시",
             "남양주시", "안양시", "평택시", "시흥시", "파주시", "의정부시",
             "김포시", "광주시", "광명시", "군포시", "하남시", "오산시", "양주시",
             "이천시", "구리시", "안성시", "포천시", "의왕시", "양평군", "여주시",
             "동두천시", "과천시", "가평군", "연천군"],
    "강원": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시",
             "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군",
             "양구군", "인제군", "고성군", "양양군"],
    "충북": ["청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군",
             "진천군", "괴산군", "음성군", "단양군"],
    "충남": ["천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시",
             "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군",
             "태안군"],
    "전북": ["전주시", "군산시", "익산시", "정읍시", "남원시", "김제시", "완주군",
             "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군"],
    "전남": ["목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군",
             "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군",
             "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군",
             "신안군"],
    "경북": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시",
             "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군",
             "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군",
             "울릉군"],
    "경남": ["창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시",
             "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군",
             "산청군", "함양군", "거창군", "합천군"],
    "제주": ["제주시", "서귀포시"]
}

############################
# 3) Streamlit 기본 UI
############################
st.title("💬 맞춤형 채용 공고 추천 챗봇")

############################
# '맞춤형 채용 공고 추천' UI
############################
if st.session_state["selected_tab"] == "job_recommendation":
    st.subheader("👍 지원자님의 요청 사항에 맞는 공고들을 추천해드려요.")
    st.subheader("""
    📌 **채팅 입력시 안내사항**

    입력하실 때 불확실하거나 모호한 부분이 있어도 걱정하지 마세요!  
    해당 내용은 생략하셔도 괜찮으며, 제공해주신 정보만으로도 최적의 채용 공고를 추천해드릴게요.
    """)

    # (1) 지원 직무(공고제목)
    job_title = st.text_input("1️⃣ 지원하고자 하는 직무를 작성해주세요.", placeholder="예) 데이터 분석가")
    # ※ '공고제목 중요도'는 더 이상 받지 않음

    # (2) 경력
    experience = st.slider("2️⃣ 지원하고자 하는 분야와 관련된 경력(근무 연수)을 선택해주세요.", 0, 20, 0)

    # (3) 근무 시/도 선택
    if st.session_state["selected_sido"] == ["전체"]:
        sido_options = ["전체"]
        default_vals = ["전체"]
    else:
        sido_options = ["전체"] + list(location_dict.keys())
        default_vals = st.session_state["selected_sido"]

    def update_sido_selection():
        current_sido = st.session_state["selected_sido_widget"]
        if "전체" in current_sido:
            st.session_state["selected_sido"] = ["전체"]
        else:
            st.session_state["selected_sido"] = [s for s in current_sido if s in location_dict]

    st.multiselect(
        "3️⃣ 원하시는 근무 위치(시/도)를 선택해주세요.",
        options=sido_options,
        default=default_vals,
        key="selected_sido_widget",
        on_change=update_sido_selection
    )

    # (4) 시/군/구 선택
    selected_sigungu = []
    if st.session_state["selected_sido"] == ["전체"]:
        for sido_key, sigungu_list in location_dict.items():
            if sido_key == "세종":
                selected_sigungu.append("세종")
            else:
                for sg in sigungu_list:
                    selected_sigungu.append(f"{sido_key} {sg}")
    else:
        for sido in st.session_state["selected_sido"]:
            if sido == "세종":
                selected_sigungu.append("세종")
                continue

            sigungu_key = f"selected_sigungu_{sido}"
            widget_key = f"{sigungu_key}_widget"

            if sigungu_key not in st.session_state:
                st.session_state[sigungu_key] = []

            def make_sigungu_callback(sido_=sido):
                def _cb():
                    current_ = st.session_state[widget_key]
                    if "전체" in current_:
                        st.session_state[sigungu_key] = ["전체"]
                    else:
                        st.session_state[sigungu_key] = [
                            sg for sg in current_ if sg in location_dict[sido_]
                        ]
                return _cb

            if st.session_state[sigungu_key] == ["전체"]:
                this_options = ["전체"]
                this_defaults = ["전체"]
            else:
                this_options = ["전체"] + location_dict[sido]
                this_defaults = st.session_state[sigungu_key]

            st.multiselect(
                f"📍 {sido}의 시/군/구를 선택해주세요.",
                options=this_options,
                default=this_defaults,
                key=widget_key,
                on_change=make_sigungu_callback(sido)
            )

        for sido in st.session_state["selected_sido"]:
            if sido == "세종":
                continue
            sigungu_key = f"selected_sigungu_{sido}"
            if sigungu_key not in st.session_state:
                continue

            if st.session_state[sigungu_key] == ["전체"]:
                for sg in location_dict[sido]:
                    selected_sigungu.append(f"{sido} {sg}")
            else:
                for sg in st.session_state[sigungu_key]:
                    selected_sigungu.append(f"{sido} {sg}")

    # (5) 원하는 업무(주요업무)
    job_task = st.text_area("4️⃣ 원하시는 업무를 작성해주세요.", placeholder="예) 저는 데이터 분석 및 시각화를 하고 싶어요.")
    job_task_importance = (
        st.slider("⭐️ 중요도", 1, 5, 3, key="job_task_importance") if job_task else None
    )

    # (6) 본인의 스킬 및 활용 가능한 툴 (자격요건 및 우대사항)
    job_skills = st.text_area("5️⃣ 본인의 스킬 및 활용 가능한 툴을 작성해주세요.", placeholder="예) Python과 SQL을 잘해요.")
    job_skills_importance = (
        st.slider("⭐️ 중요도", 1, 5, 3, key="job_skills_importance") if job_skills else None
    )

    # (7) 원하시는 혜택 및 복지
    job_benefits = st.text_area("6️⃣ 원하시는 혜택 및 복지를 작성해주세요.", placeholder="예) 유연근무가 가능했으면 좋겠어요")
    job_benefits_importance = (
        st.slider("⭐️ 중요도", 1, 5, 3, key="job_benefits_importance") if job_benefits else None
    )

    # =========================
    # 🚀 제출 버튼
    # =========================
    if st.button("🚀 답변 제출"):
        ####################################################################
        # A) 사용자 입력 구조화: 경력, 근무위치 => 하드필터
        #    (주요업무, 자격요건및우대사항, 혜택및복지) => 소프트필터
        #    (공고제목) => 별도 로직
        ####################################################################
        hard_filter_dict = {
            "경력": experience,
            "근무위치": selected_sigungu
        }

        # 소프트필터(주요업무, 자격요건및우대사항, 혜택및복지)만 dict에 담음
        # 공고제목은 여기 안 넣음
        soft_filters = []
        if job_task and job_task_importance is not None:
            soft_filters.append(("주요업무", [job_task.strip()], job_task_importance))
        if job_skills and job_skills_importance is not None:
            # "자격요건및우대사항" 컬럼에 매칭
            soft_filters.append(("자격요건및우대사항", [job_skills.strip()], job_skills_importance))
        if job_benefits and job_benefits_importance is not None:
            soft_filters.append(("혜택및복지", [job_benefits.strip()], job_benefits_importance))

        total_importance = sum([f[2] for f in soft_filters])
        soft_filter_dict = {}
        if total_importance > 0:
            for col_name, kw_list, imp in soft_filters:
                weight = round(imp / total_importance, 4)
                soft_filter_dict[col_name] = {
                    "가중치": weight,
                    "조건": kw_list
                }
                
        user_input_json = {"soft_filter": soft_filter_dict}

        # job_title은 별도로 보관
        job_title_input = job_title.strip()
        
        print("=== [Debug] user_input_json ===")
        print(user_input_json)
        print("=== [Debug] hard_filter ===")
        print(hard_filter_dict)
        print("=== [Debug] soft_filter_dict ===")
        print(json.dumps(soft_filter_dict, ensure_ascii=False, indent=2))
        print("=== [Debug] job_title_input ===", job_title_input)
        print("===========================================")

        ####################################################################
        # B) ChromaDB 연결 및 임베딩 모델 로드
        ####################################################################
        # 임베딩 모델 타입: "bge" 또는 "jina"
        embedding_model_type = "bge"  # 예: bge 사용

        if embedding_model_type == "bge":
            db_path = "./chroma_db_bge"
            bge_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=False,   # CPU라면 False 권장
                device="cpu"
            )
            def embed_with_model(text: str):
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
        else:
            db_path = "./chroma_db_jina"
            jina_model = JinaEmbeddingModel(
                model_name="jinaai/jina-embeddings-v3",
                use_fp16=False,   # CPU
                task="text-matching"
            )
            def embed_with_model(text: str):
                if not text.strip():
                    text = " "
                out = jina_model.encode(
                    [text],
                    max_length=1024
                )
                return out["dense_vecs"][0]

        def cosine_similarity(vec1, vec2):
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))

        # ChromaDB Client & Collection
        client_chroma = chromadb.PersistentClient(path=db_path)
        collection_name = "job_postings_collection"
        try:
            collection = client_chroma.get_collection(collection_name)
        except:
            st.error("ChromaDB 컬렉션이 존재하지 않아요.")
            st.stop()

        ####################################################################
        # C) 하드필터 (경력, 근무위치) -> ChromaDB Where 조건
        ####################################################################
        hard_exp = float(hard_filter_dict["경력"])
        hard_locs = hard_filter_dict["근무위치"]

        and_conditions = []
        # (1) 경력 조건
        and_conditions.append({"경력": {"$lte": hard_exp}})

        # (2) 근무위치 조건
        if "전체" not in hard_locs and len(hard_locs) > 0:
            and_conditions.append({"근무위치": {"$in": hard_locs}})

        if len(and_conditions) == 1:
            where_clause = and_conditions[0]
        elif len(and_conditions) > 1:
            where_clause = {"$and": and_conditions}
        else:
            where_clause = {}

        filtered_docs = collection.get(
            where=where_clause,
            include=["embeddings", "metadatas"],
            limit=999999
        )

        if len(filtered_docs["ids"]) == 0:
            st.warning("경력 및 근무위치 조건을 만족하는 공고가 없어요.")
            st.stop()

        ####################################################################
        # D) 공고제목 + 나머지 소프트필터 로직 분기
        ####################################################################
        # - Case A: job_title만 있고 (job_task, job_skills, job_benefits)는 없음
        # - Case B: job_title + (job_task 또는 job_skills 또는 job_benefits 중 하나 이상) 존재
        # - Case C: job_title이 없고, 나머지 소프트필터가 있는 경우
        # - Case D: job_title이 없고, 나머지 소프트필터도 없는 경우
        ####################################################################

        has_job_title = bool(job_title_input)
        has_job_task = bool(job_task.strip())
        has_job_skills = bool(job_skills.strip())
        has_job_benefits = bool(job_benefits.strip())

        soft_filter_count = sum([has_job_task, has_job_skills, has_job_benefits])

        # ================================
        # D-1) 유틸: 최종 공고 표시 함수
        # ================================
        def display_partial_text(label: str, text: str, char_limit=100):
            if not text or pd.isna(text):
                st.markdown(f"**{label}:** 내용이 게시되어 있지 않아요!")
                return
            text = apply_indentation(text)
            if len(text) <= char_limit:
                st.markdown(f"**{label}:**")
                st.text(text)
            else:
                truncated = text[:char_limit] + "..."
                st.markdown(f"**{label}:**")
                st.text(truncated)
                with st.expander("상세 보기"):
                    st.text(text)

        def apply_indentation(text):
            lines = text.split('\n')
            indented_lines = []
            for line in lines:
                if any(line.strip().startswith(marker) for marker in INDENTATION_MARKERS):
                    indented_lines.append(f"    {line.strip()}")
                else:
                    indented_lines.append(line)
            return '\n'.join(indented_lines)

        def show_job_postings(final_df):
            for idx, (_, row) in enumerate(final_df.iterrows(), start=1):
                st.markdown(f"### Top {idx}: {row['공고제목']}")
                st.markdown(f"**회사명:** {row['회사명']}")
                display_partial_text("주요 업무", row.get("주요업무", ""))
                display_partial_text("자격 요건", row.get("자격요건", ""))
                display_partial_text("우대 사항", row.get("우대사항", ""))
                display_partial_text("혜택 및 복지", row.get("혜택및복지", ""))
                st.markdown(f"**근무 위치:** {row.get('근무위치','')}")
                exp_str = "신입" if int(row.get("경력",0)) == 0 else f"{int(row.get('경력',0))}년 이상"
                st.markdown(f"**경력:** {exp_str}")
                st.markdown(f"**최종 점수:** {row.get('최종점수','0.0')}")
                url_val = row.get("공고상세url", "")
                if pd.notna(url_val) and url_val:
                    st.markdown(f"**URL:** [바로가기]({url_val})")
                st.markdown("---")

        # ==============================================
        # D-2) “소프트필터” (주요업무, 자격요건및우대사항, 혜택및복지) 계산 함수
        # ==============================================
        def calc_soft_filter_scores(docs, user_filter_dict):
            """
            docs: 하드필터를 통과하고 (공고제목이든 주요업무든 전부) 가져온 ChromaDB 문서들.
            user_filter_dict: {"주요업무": {...}, "자격요건및우대사항": {...}, "혜택및복지": {...}}
            (가중치, 조건) 구조
            """
            # 1) 각 필드별 사용자 임베딩
            keyword_embeddings = {}
            for col_type, info in user_filter_dict.items():
                emb_list = []
                for kw in info["조건"]:
                    emb_list.append(embed_with_model(kw))
                keyword_embeddings[col_type] = emb_list

            # 2) job_id별 raw 유사도
            sim_raw = {}
            for i, doc_id in enumerate(docs["ids"]):
                emb = np.array(docs["embeddings"][i], dtype=np.float32)
                meta = docs["metadatas"][i]
                j_id = meta["공고id"]
                t = meta["type"]

                if j_id not in sim_raw:
                    sim_raw[j_id] = {}

                # 이 doc의 type이 user_filter_dict에 있다면
                if t in user_filter_dict:
                    kw_embs = keyword_embeddings[t]
                    if len(kw_embs) == 0:
                        raw_sim = 0.0
                    else:
                        # 사용자 입력문 여러 개면 평균
                        scores = []
                        for kw_vec in kw_embs:
                            s = cosine_similarity(emb, kw_vec)
                            scores.append(s)
                        raw_sim = np.mean(scores) if scores else 0.0
                    sim_raw[j_id][t] = raw_sim

            # 3) 최종 점수 (가중합)
            final_scores = {}
            for j_id in sim_raw.keys():
                score_sum = 0.0
                for doc_type, info in user_filter_dict.items():
                    raw_val = sim_raw[j_id].get(doc_type, 0.0)
                    weight = info["가중치"]
                    score_sum += raw_val * weight
                final_scores[j_id] = score_sum

            return final_scores

        # ======================================================
        # D-3) 하드필터 통과한 문서들 중에서 job_id 리스트 추출
        # ======================================================
        all_job_ids = []
        for meta in filtered_docs["metadatas"]:
            j_id = meta["공고id"]
            if j_id not in all_job_ids:
                all_job_ids.append(j_id)

        # ======================================================
        # D-4) 상황별 분기
        # ======================================================
        if has_job_title and soft_filter_count == 0:
            # --------------------------------------------------
            # Case A: 공고제목 "단독"
            # => "경력,근무위치" 하드필터 통과한 docs 중 "type=공고제목"과의 유사도 => top5
            # --------------------------------------------------
            title_vec = embed_with_model(job_title_input)

            # type이 "공고제목"인 문서만 골라서 유사도 계산
            doc_scores = {}
            for i, doc_id in enumerate(filtered_docs["ids"]):
                meta = filtered_docs["metadatas"][i]
                t = meta["type"]
                j_id = meta["공고id"]
                if t == "공고제목":
                    emb = np.array(filtered_docs["embeddings"][i], dtype=np.float32)
                    sim = cosine_similarity(title_vec, emb)
                    doc_scores[j_id] = sim

            if not doc_scores:
                st.warning("공고제목 임베딩을 계산했지만, 해당 타입 문서가 없어요.")
                st.stop()

            # 점수 기준 top5
            sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:5]

            # all_raw.xlsx에서 해당 공고 세부 정보를 가져오기
            df_all = pd.read_excel("./all_raw.xlsx")
            df_all["공고id"] = df_all["공고id"].astype(str)

            top_df = df_all[df_all["공고id"].isin(sorted_ids)].copy()
            top_df["최종점수"] = top_df["공고id"].apply(lambda x: round(doc_scores.get(str(x), 0.0), 4))
            top_df = top_df.sort_values("최종점수", ascending=False)

            if len(top_df) == 0:
                st.warning("공고제목 유사도 기반 추천 결과가 없어요.")
                st.stop()

            st.success("🔎 작성하신 직무 기반 상위 5개 공고를 보여드려요!")
            show_job_postings(top_df)

        elif has_job_title and soft_filter_count > 0:
            # --------------------------------------------------
            # Case B: 공고제목 + (주요업무 or 자격요건 or 혜택 등) 1개 이상
            # => 2단계 하드필터(공고제목 유사도 ≥ 0.6) -> 소프트필터
            # --------------------------------------------------
            title_vec = embed_with_model(job_title_input)

            # (1) 공고제목 유사도가 일정 threshold 이상인 공고만 남김
            pass_ids = []
            for i, doc_id in enumerate(filtered_docs["ids"]):
                meta = filtered_docs["metadatas"][i]
                t = meta["type"]
                j_id = meta["공고id"]
                if t == "공고제목":
                    emb = np.array(filtered_docs["embeddings"][i], dtype=np.float32)
                    sim = cosine_similarity(title_vec, emb)
                    if sim >= 0.4:
                        if j_id not in pass_ids:
                            pass_ids.append(j_id)

            if not pass_ids:
                st.warning("직무 조건의 threshold를 만족하는 공고가 없습니다.")
                st.stop()

            # (2) pass_ids에 속한 문서만 다시 모아서 -> 소프트필터(주요업무, 자격요건, 혜택 등) 계산
            pass_docs_ids = []
            pass_docs_embeddings = []
            pass_docs_metas = []

            for i, doc_id in enumerate(filtered_docs["ids"]):
                meta = filtered_docs["metadatas"][i]
                j_id = meta["공고id"]
                if j_id in pass_ids:
                    pass_docs_ids.append(doc_id)
                    pass_docs_embeddings.append(filtered_docs["embeddings"][i])
                    pass_docs_metas.append(meta)

            # 재구성한 docs dict
            pass_docs = {
                "ids": pass_docs_ids,
                "embeddings": pass_docs_embeddings,
                "metadatas": pass_docs_metas
            }

            if len(soft_filter_dict) == 0:
                # 만약 소프트필터가 하나도 없다면 => 통과한 공고 중 상위 5개 그냥 반환
                # (하지만 이미 has_job_title and soft_filter_count>0 라서 여기선 발생하지 않겠지만 혹시 모를 대비)
                job_ids = []
                for meta in pass_docs["metadatas"]:
                    j_id = meta["공고id"]
                    if j_id not in job_ids:
                        job_ids.append(j_id)
                top5_ids = job_ids[:5]
                df_all = pd.read_excel("./all_raw.xlsx")
                df_all["공고id"] = df_all["공고id"].astype(str)
                filtered_df = df_all[df_all["공고id"].isin(top5_ids)].copy()
                filtered_df["최종점수"] = 0.0
                show_job_postings(filtered_df)
                st.stop()
            else:
                # 소프트필터 계산
                final_scores = calc_soft_filter_scores(pass_docs, soft_filter_dict)
                if not final_scores:
                    st.warning("소프트필터 점수를 계산할 문서가 없어요.")
                    st.stop()
                # top5
                sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:5]
                if not sorted_ids:
                    st.warning("소프트필터를 만족하는 상위 공고가 없어요.")
                    st.stop()

                df_all = pd.read_excel("./all_raw.xlsx")
                df_all["공고id"] = df_all["공고id"].astype(str)
                top_df = df_all[df_all["공고id"].isin(sorted_ids)].copy()
                top_df["최종점수"] = top_df["공고id"].apply(lambda x: round(final_scores.get(str(x), 0.0), 4))
                top_df = top_df.sort_values("최종점수", ascending=False)

                st.success("🔎 맞춤형 공고 상위 5개를 보여드려요!")
                show_job_postings(top_df)

        else:
            # --------------------------------------------------
            # 공고제목이 없는 경우 -> Case C, D
            # Case C: 소프트필터(주요업무,자격요건,혜택) 존재
            # Case D: 소프트필터도 없음
            # --------------------------------------------------
            if len(soft_filter_dict) == 0:
                # Case D: 소프트필터 전무
                job_ids = []
                for meta in filtered_docs["metadatas"]:
                    j_id = meta["공고id"]
                    if j_id not in job_ids:
                        job_ids.append(j_id)
                top5 = job_ids[:5]

                df_all = pd.read_excel("./all_raw.xlsx")
                df_all["공고id"] = df_all["공고id"].astype(str)
                filtered_df = df_all[df_all["공고id"].isin(top5)].copy()
                filtered_df["최종점수"] = 0.0
                show_job_postings(filtered_df)
            else:
                # Case C: 공고제목은 없고, 소프트필터 존재
                final_scores = calc_soft_filter_scores(filtered_docs, soft_filter_dict)
                if not final_scores:
                    st.warning("소프트필터 점수를 계산할 문서가 없어요.")
                    st.stop()
                sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:5]
                if not sorted_ids:
                    st.warning("소프트필터 결과, 상위 공고가 없어요.")
                    st.stop()

                df_all = pd.read_excel("./all_raw.xlsx")
                df_all["공고id"] = df_all["공고id"].astype(str)
                top_df = df_all[df_all["공고id"].isin(sorted_ids)].copy()
                top_df["최종점수"] = top_df["공고id"].apply(lambda x: round(final_scores.get(str(x), 0.0), 4))
                top_df = top_df.sort_values("최종점수", ascending=False)

                st.success("🔎 맞춤형 공고 상위 5개를 보여드려요!")
                show_job_postings(top_df)

        ######################################################
        # 추가: 추천 사유를 생성하는 함수 정의 (공고제목은 제외)
        ######################################################
        def generate_recommendation_rationale(user_input_json, top_df):
            provided_fields = [key for key in user_input_json["soft_filter"].keys()]
        
            # 프롬프트 초기 구성
            prompt = "아래는 사용자가 입력한 소프트 필터 정보와 추천된 Top 5 공고의 주요 내용입니다.\n\n"
        
            # 사용자 입력 (소프트 필터)
            prompt += "사용자 입력 (소프트 필터):\n"
            for key in provided_fields:
                if key == "자격요건및우대사항":
                    prompt += f"- 자격요건및우대사항 (자격요건 및 우대사항 모두 해당): {user_input_json['soft_filter'][key]['조건']}\n"
                else:
                    prompt += f"- {key}: {user_input_json['soft_filter'][key]['조건']}\n"
        
            # 추천된 채용 공고 내용 - Top 순서대로
            prompt += "\n추천된 채용 공고 내용:\n"
            for i, (_, row) in enumerate(top_df.iterrows(), start=1):
                prompt += f"Top {i}: **{row['공고제목']}**\n"
                if "주요업무" in provided_fields:
                    prompt += f"  - **주요업무:** {row['주요업무']}\n\n"
                if "자격요건및우대사항" in provided_fields:
                    prompt += f"  - **자격요건:** {row.get('자격요건', '')}\n\n"
                    prompt += f"  - **우대사항:** {row.get('우대사항', '')}\n\n"
                if "혜택및복지" in provided_fields:
                    prompt += f"  - **혜택및복지:** {row['혜택및복지']}\n\n"
                prompt += "\n---\n\n"
        
            # 사용자가 입력한 필드만 설명하도록 요청
            fields_explanation = []
            if "주요업무" in provided_fields:
                fields_explanation.append("주요업무")
            if "자격요건및우대사항" in provided_fields:
                fields_explanation.append("자격요건")
                fields_explanation.append("우대사항")
            if "혜택및복지" in provided_fields:
                fields_explanation.append("혜택및복지")
            fields_text = ", ".join(fields_explanation)
        
            prompt += (
                "위 내용을 기반으로, 각 추천 공고에 대해 사용자가 입력한 소프트 필터 항목 중 "
                f"[{fields_text}]에 해당하는 부분이 공고 내용에서 어떻게 나타나는지 아래 형식으로 설명해 주세요.\n\n"
                "형식 (마크다운 형식):\n"
                "🔷**Top 1: [공고제목]**\n\n"
            )
            if "주요업무" in provided_fields:
                prompt += " ▪️ **주요업무:** <설명>\n\n"
            if "자격요건및우대사항" in provided_fields:
                prompt += " ▪️ **자격요건:** <설명>\n\n"
                prompt += " ▪️ **우대사항:** <설명>\n\n"
            if "혜택및복지" in provided_fields:
                prompt += " ▪️ **혜택및복지:** <설명>\n\n"
        
            # ▼▼▼ 변경/추가된 부분 ▼▼▼
            prompt += (
                "단, 사용자가 입력하지 않은 항목은 아예 설명에서 생략해 주세요. "
                "또한, 해당 항목이 공고 내용에서 명확하게 나타나지 않는 경우, '해당 항목과 관련된 내용이 명확하게 나타나지 않아요'라고 간단하게 언급해 주세요.\n\n"
                "**중요**: Top 1부터 Top 5까지를 절대로 생략하지 말고 전부 별도로 설명해 주세요. "
                "'이하 생략', '...' 등의 요약 표현 없이, 각 공고를 모두 구체적으로 작성해 주시기 바랍니다."
            )
            # ▲▲▲ 변경/추가된 부분 ▲▲▲
        
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant who explains job recommendation rationale based on user input "
                                "and job posting data in markdown format. Do not fabricate explanations if the user's input "
                                "is not clearly supported by the job posting content."
                            )
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5
                )
                explanation = response.choices[0].message.content
            except Exception as e:
                explanation = f"추천 사유를 생성하는 데 오류가 발생했어요: {e}"
        
            return explanation

        
        ######################################################
        # 추가: 로딩 메시지와 함께 추천 사유 생성 및 출력
        ######################################################
        loading_msg = st.empty()
        loading_msg.markdown("#### ⏳공고 추천 이유를 알려드릴게요. 잠시만 기다려주세요☺️⌛")

        explanation = generate_recommendation_rationale(user_input_json, top_df)

        loading_msg.empty()

        st.markdown("### 공고 추천 이유")
        st.write(explanation)