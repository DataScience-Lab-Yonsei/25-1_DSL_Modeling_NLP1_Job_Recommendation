import openai
import google.generativeai as genai
import pandas as pd
import json
from tqdm import tqdm
import time
import re

############################
# API 설정
############################
# Gemini API 설정
GOOGLE_API_KEY = "개인_API_key" 
genai.configure(api_key=GOOGLE_API_KEY)

############################
# 0) 전역 DataFrame 준비
############################
coverage_result_columns = ["model", "num", "rank", "주요업무", "자격요건및우대사항", "혜택및복지"]
coverage_df = pd.DataFrame(columns=coverage_result_columns)

############################
# 1) Gemini API 이용한 키워드 추출 함수
## 입력: ./user_input.xlsx
## 출력: ./user_input_keyword.json
############################
def extract_keywords(text: str) -> list:
    """ GPT 기반 키워드 추출 함수 """
    if not text.strip():
        return []

    system_prompt = (
        """
        ## Role
        You are an AI assistant responsible for understanding a user’s recruitment or job-related requirements. 
        You must correct any spelling or grammatical errors in the user’s text, then extract and return the essential keywords in JSON format.

        ## Common Conditions
        1) Keywords must be concise, between 1 and 4 words in length.
        2) If there are spelling or grammatical mistakes, you should correct them naturally before extracting keywords.
        3) Avoid over-segmentation: if multiple words share the same core meaning, combine them into one keyword.
        4) The output must be in JSON format: {"keywords": ["keyword1", "keyword2"]}
        5) Provide no additional explanatory text or any output other than the JSON structure.
        6) If a term is originally from English but spelled in Korean (e.g., “마캐팅”), identify and correct the likely English origin (e.g., “Marketing”) and use that corrected form when generating the final keyword.
        7) Extract keywords in the **same language as the input**:  
           - **Korean input → Korean keywords**  
           - **English input → English keywords** 
        8) For conjunctions like “및,” “and,” or “or,” follow these rules:  
           - **Same Action/Concept:** Treat as one keyword.  
                 - Example: "머신러닝 및 딥러닝 모델 구축" → ["머신러닝 및 딥러닝 모델 구축"]  
           - **Distinct Actions/Concepts:** Separate into individual keywords.  
                 - Example: "Python 및 R 프로그래밍" → ["Python 프로그래밍", "R 프로그래밍"]  
           - **Unclear Context:** Default to one keyword.
                 - Example: "LLM 모델 파인튜닝 및 구현" → ["LLM 모델 파인튜닝 및 구현"]  

        ## Category-Specific Conditions

        ### Job Title
        - If the text refers to a job title, consider the entire phrase (even if it consists of multiple words) as one single keyword.
        - Example1: “Data Analytics Manager” → single keyword
        - Example2: "전략 컨솔터트" → "전략 컨설턴트" (after correcting spelling or grammar)

        ### Main Duties
        - If the text describes key tasks or responsibilities, extract them as separate keywords in 1–4 words.
        - Example: “analyzing business metrics,” “creating data dashboards”

        ### Qualifications / Preferred Skills
        - If the text discusses qualifications or preferred skills, extract them in concise 1–4 word phrases.
        - Example: “Python proficiency,” “Statistical modeling,” “Machine learning experience”

        ### Benefits / Welfare
        - If multiple benefit or welfare items appear in a single sentence, separate them into individual keywords.
        - Example: “Flexible working hours,” “Health insurance,” “Onsite gym access”
        """
    )

    user_prompt = (
        f" Please extract concise keywords (1–4 words) from the text below, following the system rules. \n"
        f"Text: {text} \n\n"
        f"Return the result in JSON with the key 'keywords' and no additional text."
    )

    start_time = time.time()
    print("🚀 Gemini API 호출 중... (키워드 추출)")

    try:
        # Gemini 모델 설정 (예: gemini-2.0-flash)
        model = genai.GenerativeModel('gemini-2.0-flash')
        temperature = 0  

        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(
            prompt_text,
            generation_config=genai.GenerationConfig(temperature=temperature)
        )

        response_text = response.text.strip()

        # JSON 응답 파싱
        if response_text.startswith("[") or response_text.startswith("{"):
            try:
                parsed = json.loads(response_text)
                if "keywords" in parsed and isinstance(parsed["keywords"], list):
                    return parsed["keywords"]
                else:
                    return []
            except json.JSONDecodeError:
                pass

        # 코드 블록 제거
        response_text = response_text.replace("```json", "").replace("```", "")

        end_time = time.time()
        print(f"✅ 완료! ⏱️ 실행 시간: {round(end_time - start_time, 2)}초")
        time.sleep(5)
        return response_text.strip()

    except Exception as e:
        print(f"❌ 키워드 추출 중 오류 발생: {str(e)}")
        return []

############################
# 2) Gemini API 이용한 커버리지 & 디버그 계산
############################
def evaluate_and_debug_with_gemini(user_input_json, top_df_BGE, top_df_Jina, api_call_count):
    """
    1) LLM 호출 → Coverage 템플릿(규격) + Debug Info(키워드별 MATCH/NO MATCH + (Matched/Total))
    2) Debug Info에서 (# Matched / # Total) = coverage 비율 계산
    3) coverage_results_debug.csv에 저장
    """
    global coverage_df

    iteration = user_input_json.get("num", 1) # 디버그 블록 부분은 그냥 텍스트로 파싱해서 print() (또는 필요시 저장)
    
    # 만약 15회마다 60초 쿨다운이 필요하다면:
    if api_call_count > 0 and api_call_count % 15 == 0:
        print("⏳ 유사도 점수 계산 중 60초 대기 중...")
        time.sleep(60)

    # --------------------------
    # (1) System Prompt
    # --------------------------
    # "No additional text"이라는 문구가 있지만,
    #   - "커버리지 템플릿" 뒤에는 "DEBUG_INFO:" 블록을 허용한다.
    #   - (커버리지 템플릿 + DEBUG_INFO 블록)을 모두 출력하도록 지시.
    system_prompt = '''
You are a professional job matching evaluator.

# CRITICAL CALCULATION INSTRUCTIONS - HIGHEST PRIORITY
For each category (job_task, job_skills, job_benefits):
1. Calculate coverage ratio PRECISELY as: (Number of MATCH items) ÷ (Total number of items in category)
2. Example:
   - If there are 3 total job_task items and 2 are marked as MATCH(1.0), then:
   - job_task ratio = 2/3 = 0.6666... (NOT 1.0)
3. Never round up to 1.0 - use the exact division result
4. Always use all items in the denominator, whether they match or not

CALCULATION ALGORITHM:
def calculate_coverage(category_items):
total_items = len(category_items)  # All items in category
matched_items = sum(1 for item in category_items if item['status'] == 'MATCH')
ratio = matched_items / total_items  # Must perform actual division
return ratio

# IMPORTANT:
We want you to produce TWO parts in your final answer:

1) The Coverage Template Block (EXACT format, no extra text):
   num_{iteration} = {
     bge_{iteration}_rank_1 = {주요업무 = X, 자격요건및우대사항 = Y, 혜택및복지 = Z},
     bge_{iteration}_rank_2 = {...},
     bge_{iteration}_rank_3 = {...},
     bge_{iteration}_rank_4 = {...},
     bge_{iteration}_rank_5 = {...},
     jina_{iteration}_rank_1 = {...},
     jina_{iteration}_rank_2 = {...},
     jina_{iteration}_rank_3 = {...},
     jina_{iteration}_rank_4 = {...},
     jina_{iteration}_rank_5 = {...}
   }

   Where X, Y, Z are coverage ratios (floats from 0.0 to 1.0). No extra lines or text.

2) The Debug Info Block (after the coverage block). 
   - Begin with the line "DEBUG_INFO:" on its own, then list debug lines. Example:
     [Debug for BGE rank=1]
     - job_task('키워드'): MATCH(1.0) → "이유"
     - job_skills('키워드'): NO MATCH(0.0) → "이유"
     ...
     - **job_task = (X / {주요업무 키워드 개수})** 
     - **job_skills = (Y / {자격요건및우대사항 키워드 개수})** 
     - **job_benefits = (Z / {혜택및복지 키워드 개수})**
    - Keep it concise, 1-2 lines per keyword describing how you decided MATCH or NO MATCH.

# Coverage Evaluation Criteria
1) Take the user's query (in JSON), which contains keywords for certain fields and the "num" field for the iteration number.
2) For each job post (Top 5 from BGE and Top 5 from Jina), focus only on these 3 fields:
   - "주요업무"
   - "자격요건및우대사항"
   - "혜택및복지"

3) For each field, compute coverage ratio as follows:
   Coverage Ratio = (# Matched Keywords) / (# Total Keywords in user's query for that field)
   - The numerator is the COUNT of keywords marked as MATCH (1.0)
   - The denominator is the TOTAL COUNT of keywords in the user's query for that field
   - Each keyword contributes exactly 0 or 1 to the numerator (no partial counting)

4) **Enhanced Semantic Similarity Rule for Matching Keywords**
   - Allow for **conceptually related**, **contextually similar**, or **domain-specific** keyword matches.
   - Prioritize **semantic similarity** over exact matching.
   - **Partial matches are NOT allowed.** However, if two terms share over 70% semantic similarity or serve the same purpose in context, treat them as MATCH.
     - Example: "사업 개발" ↔ "비즈니스 성장 전략" (✅ MATCH)
     - Example: "파트너십 개발" ↔ "협력 기회 확대" (✅ MATCH)

5) **Flexible Matching Guidelines for ALL COLUMNS**
   - Recognize **broader concepts**, **synonyms**, and **job-specific terminology** as matches.
   - Allow flexible interpretation for all 3 categories: **주요업무**, **자격요건및우대사항**, **혜택및복지**.
   - **Partial matches are NOT allowed.**
   - Match keywords that are **broader in concept** but still relevant.  
     - Example: "경영 컨설턴트" ↔ "전략 컨설턴트" (✅ Match)  
     - Example: "기업 경영 분석" ↔ "비즈니스 성과 분석" (✅ Match)  
     - Example: "재무 분석" ↔ "재무 리포트 작성" (✅ Match)  
   - Consider **role-related synonyms** or job-specific terminology.  
     - Example: "브랜드 전략 기획" ↔ "브랜드 마케팅 기획" (✅ Match)  
     - Example: "광고 캠페인 운영" ↔ "마케팅 이벤트 기획" (✅ Match)  
   - Match **supportive skills or experience** with relevant technical keywords.  
     - Example: "데이터 분석 경험" ↔ "SQL, 파이썬 기반 데이터 분석" (✅ Match)  
     - Example: "컨설팅 프로젝트 경험" ↔ "프로젝트 매니지먼트" (✅ Match)  
   - Allow for flexible interpretation in matching 복지 혜택.  
     - Example:
       - "자율 출퇴근 제도" ↔ "유연 근무" (✅ Match)
       - "헬스장 지원" ↔ "헬스 멤버십 제공" (✅ Match)
       - "식대 지원" ↔ "점심 제공" (✅ Match)
       - "연차 제도" ↔ "연차 유급 휴가" (✅ Match)

6) Matching Rules for Accuracy:
   - Count each keyword only ONCE per field (no duplicates).  
   - Prevent inflated scores by limiting **redundant matches** to a single count.
   - Identify synonyms, semantically equivalent expressions, or translated terms as matches.  
   - Avoid matching unrelated terms or keywords that distort the intended meaning.
     - "경영" ↔ "기업 경영 분석" (❌ No Match)
     - "데이터 삭제" ↔ "데이터 분석" (❌ No Match)

7) **Keyword Expansion for Comprehensive Matching**
   - Broaden keyword matching to include relevant concepts and skills.
     - Example: "머신러닝" ↔ "AI 모델 구축" (✅ Match)  
     - Example: "브랜드 전략" ↔ "브랜드 관리" (✅ Match)  
   - Consider **industry-specific language** as valid matches.
     - Example: "SQL" ↔ "데이터베이스" (✅ Match)  
     - Example: "전략 기획" ↔ "전략적 의사결정" (✅ Match)  

8) For 자격요건및우대사항:  
   **Use the newly created column '자격요건및우대사항' in the job post to determine matching.**  
   In other words, if the user has N total keywords under "자격요건및우대사항," then count how many of these keywords appear in the combined '자격요건및우대사항' column of the job post.  
   Summing these matches yields the coverage numerator; dividing by N yields the coverage ratio for "자격요건및우대사항."  
   (Note that each keyword is counted only once per field, even if it appears multiple times in the text.)

9) For each post (rank=1..5 in BGE and Jina), calculate the coverage ratio for each of these 3 fields, then fill in the values X, Y, Z in the template accordingly.

10) **Coverage Ratio Calculation - CRITICAL**:
   - The denominator MUST always equal the user's total keywords for that field.
   - The numerator is the COUNT of items marked as MATCH(1.0).
   - Format example: job_task = (2.0 / 3)
     - Use numbers only, no words like "# Matched Keywords."
   - The final coverage ratio in the Coverage Template must exactly match these counts.
   - IMPORTANT: A ratio of 2/3 should be calculated as 0.6666... not as 1.0

# NO Extra Explanation beyond these two blocks
1) The coverage block
2) "DEBUG_INFO:" block

Any other text is not allowed.
'''

    # --------------------------
    # (2) User Prompt
    # --------------------------
    iteration = user_input_json.get("num", 1)
    
    user_prompt = f"""
    User's query (JSON): {user_input_json}
    The iteration is #{iteration}.
    
        
    Below are the top 5 job posts from BGE and top 5 job posts from Jina (all fields). 
    Calculate coverage ratio only for "주요업무", "자격요건및우대사항", "혜택및복지".
    Then produce your final answer strictly in the required template.
    """

    # 시행번호에 맞게 BGE/JINA 필터
    filtered_bge = top_df_BGE[top_df_BGE['시행'] == iteration].sort_values(by='rank')
    filtered_jina = top_df_Jina[top_df_Jina['시행'] == iteration].sort_values(by='rank')

    # --------------------------
    # (2) User Prompt
    # --------------------------
    user_prompt = f"User's query (JSON): {user_input_json}\nThe iteration is #{iteration}.\n\n"

    # 시행번호에 맞게 BGE/JINA 필터
    filtered_bge = top_df_BGE[top_df_BGE['시행'] == iteration].sort_values(by='rank')
    filtered_jina = top_df_Jina[top_df_Jina['시행'] == iteration].sort_values(by='rank')

    # BGE posts
    doc_bge_str = "BGE posts:\n"
    for idx, row in filtered_bge.iterrows():
        doc_bge_str += (
             f"[Rank={row.get('rank','')}] "
             f"주요업무: {row.get('주요업무','')}\n"
             f"자격요건및우대사항: {row.get('자격요건및우대사항','')}\n"
             f"혜택및복지: {row.get('혜택및복지','')}\n\n"
        )

    # Jina posts
    doc_jina_str = "Jina posts:\n"
    for idx, row in filtered_jina.iterrows():
        doc_jina_str += (
             f"[Rank={row['rank']}] "
             f"주요업무: {row.get('주요업무','')}\n"
             f"자격요건및우대사항: {row.get('자격요건및우대사항','')}\n"
             f"혜택및복지: {row.get('혜택및복지','')}\n\n"
        )
    
    final_user_prompt = f"{user_prompt}\n\n{doc_bge_str}\n{doc_jina_str}\n"

    # --------------------------
    # (3) LLM 호출
    # --------------------------
    print("🚀 Gemini API 호출 (Coverage + Debug) ...")
    start_time = time.time()
    
    # Gemini 호출
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt_text = f"{system_prompt}\n\n{final_user_prompt}"
        response = model.generate_content(
            prompt_text, generation_config=genai.GenerationConfig(temperature=0.0)
        )
        response_text = response.text.strip()

        # Coverage 템플릿 + Debug Info 분할
        coverage_part, debug_part = split_coverage_and_debug(response_text)

        # Debug Info에서 Coverage 비율 추출
        debug_info = format_debug_info_with_ratios(debug_part, user_input_json)

        # Coverage CSV 저장
        save_coverage_from_debug(debug_info, iteration)

        # Debug 파일로 저장
        debug_filename = f"debug_log_{iteration}.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump({"debug_info": debug_info}, f, ensure_ascii=False, indent=2)
        print(f"✅ 디버그 로그 저장 완료: {debug_filename}")

        end_time = time.time()
        print(f"✅ 완료! ⏱️ 실행 시간: {round(end_time - start_time, 2)}초")

        response_text = coverage_part + "\nDEBUG_INFO:\n" + debug_part
        return response_text

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        if "429" in str(e):
            print("⏳ 429 오류 발생, 60초 대기 후 재시도...")
            time.sleep(60)  # 429 오류 발생 시 1분 대기 후 재시도
            return evaluate_and_debug_with_gemini(user_input_json, top_df_BGE, top_df_Jina, api_call_count)  
        return "❌ 최종 오류: 데이터 호출 실패"

############################
# 3) Coverage와 Debug 분리 함수
############################
def split_coverage_and_debug(full_text: str):
    """
    LLM 응답에서:
    1) Coverage 템플릿
    2) Debug Info
    를 분리해 반환
    """
    if "DEBUG_INFO:" in full_text:
        parts = full_text.split("DEBUG_INFO:", 1)
        coverage_part = parts[0].strip()
        debug_part = parts[1].strip()
    else:
        coverage_part = full_text
        debug_part = ""

    return coverage_part, debug_part

#############################
# 4-1) Debug 파싱 함수
###############################
def format_debug_info_with_ratios(debug_block: str, user_input_json: dict) -> dict:
    """
    1) Debug Block에서 job_task = (X / Y) 등 비율을 파싱하여 정확히 반영
    2) MATCH / NO MATCH 정보를 함께 저장
    """
    debug_info = {}
    current_rank = None
    
    # 총 키워드 개수
    total_task_keywords = len(user_input_json.get('주요업무', []))
    total_skills_keywords = len(user_input_json.get('자격요건및우대사항', []))
    total_benefits_keywords = len(user_input_json.get('혜택및복지', []))
    
    lines = debug_block.strip().split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 새로운 랭크 시작
        if "[Debug for" in line:
            if current_rank:
                # 이전 랭크에 대한 정보를 저장
                debug_info[current_rank] = {
                    "job_task": task_ratio,
                    "job_skills": skills_ratio,
                    "job_benefits": benefits_ratio,
                    "match_details": match_details
                }
            
            # 현재 랭크 정보 초기화
            current_rank = line.strip("[]")
            match_details = []
            matched_task = matched_skills = matched_benefits = 0
            task_ratio = skills_ratio = benefits_ratio = 0.0
            
            # 현재 랭크의 매칭 정보 수집
            i += 1
            collecting_details = True
            
            while i < len(lines) and collecting_details:
                detail_line = lines[i].strip()
                
                # 다음 랭크가 시작되면 중단
                if "[Debug for" in detail_line:
                    i -= 1  # 다음 반복에서 이 라인을 다시 처리하도록
                    collecting_details = False
                    continue
                
                # 비어있는 라인 건너뛰기
                if not detail_line:
                    i += 1
                    continue
                
                # 매칭 상세 정보 저장
                if "job_task(" in detail_line or "job_skills(" in detail_line or "job_benefits(" in detail_line:
                    match_details.append(detail_line)
                    if "MATCH(1.0)" in detail_line:
                        if "job_task(" in detail_line:
                            matched_task += 1
                        elif "job_skills(" in detail_line:
                            matched_skills += 1
                        elif "job_benefits(" in detail_line:
                            matched_benefits += 1
                
                # 비율 정보 파싱
                if "job_task =" in detail_line:
                    match = re.search(r'\((\d+)(?:\.\d+)?\s*/\s*(\d+)\)', detail_line)
                    if match:
                        numerator = int(match.group(1))
                        denominator = int(match.group(2))
                        if denominator > 0:
                            task_ratio = numerator / denominator
                        else:
                            task_ratio = 0.0
                
                elif "job_skills =" in detail_line:
                    match = re.search(r'\((\d+)(?:\.\d+)?\s*/\s*(\d+)\)', detail_line)
                    if match:
                        numerator = int(match.group(1))
                        denominator = int(match.group(2))
                        if denominator > 0:
                            skills_ratio = numerator / denominator
                        else:
                            skills_ratio = 0.0
                
                elif "job_benefits =" in detail_line:
                    match = re.search(r'\((\d+)(?:\.\d+)?\s*/\s*(\d+)\)', detail_line)
                    if match:
                        numerator = int(match.group(1))
                        denominator = int(match.group(2))
                        if denominator > 0:
                            benefits_ratio = numerator / denominator
                        else:
                            benefits_ratio = 0.0
                
                i += 1
            
            # 루프를 나왔을 때 i를 증가시키지 않음 (이미 위의 루프에서 증가됨)
            continue
        
        i += 1
    
    # 마지막 랭크 저장
    if current_rank:
        debug_info[current_rank] = {
            "job_task": task_ratio,
            "job_skills": skills_ratio,
            "job_benefits": benefits_ratio,
            "match_details": match_details
        }
    
    return debug_info

############################
# 4-2) Coverage 저장 함수
############################
def save_coverage_from_debug(debug_info: dict, iteration_num: int):
    """
    Debug Info를 기반으로 coverage_results_debug.csv에 저장
    """
    global coverage_df

    row_list = []
    for rank_key, coverage_dict in debug_info.items():
        if "BGE" in rank_key.upper():
            model_name = "bge"
        else:
            model_name = "jina"

        # rank= 뒤의 숫자 추출
        rank_match = re.search(r'rank=(\d+)', rank_key)
        rank_str = rank_match.group(1) if rank_match else "1"

        row_dict = {
            "model": model_name,
            "num": str(iteration_num),
            "rank": rank_str,
            "주요업무": coverage_dict["job_task"],
            "자격요건및우대사항": coverage_dict["job_skills"],
            "혜택및복지": coverage_dict["job_benefits"]
        }
        row_list.append(row_dict)

    if row_list:
        row_df = pd.DataFrame(row_list)
        coverage_df = pd.concat([coverage_df, row_df], ignore_index=True)
        coverage_df.to_csv("coverage_results.csv", index=False, encoding='utf-8-sig')
        print(f"✅ coverage_results_debug.csv 파일에 {len(row_list)}개 행 저장 완료")

############################
# 5) 사용자의 중요도를 가중치 반환 후, 최종 점수 계산 함수
############################
def calculate_finscore(coverage_df, user_data):
    """
    중요도 고려한 최종 점수 계산 함수
    """
    weights_dict = {
        item['num']: {
            'job_skills_weight': item['job_skills']['weight'],
            'job_task_weight': item['job_task']['weight'],
            'job_benefits_weight': item['job_benefits']['weight']
        }
        for item in user_data
    }

    def apply_weights(row):
        weights = weights_dict.get(row['num'], {
            'job_skills_weight': 0,
            'job_task_weight': 0,
            'job_benefits_weight': 0
        })
        row['주요업무'] = row['주요업무'] * weights['job_task_weight']
        row['자격요건및우대사항'] = row['자격요건및우대사항'] * weights['job_skills_weight']
        row['혜택및복지'] = row['혜택및복지'] * weights['job_benefits_weight']
        row['종합점수'] = row['주요업무'] + row['자격요건및우대사항'] + row['혜택및복지']
        return row

    return coverage_df.apply(apply_weights, axis=1)


############################
# 5) main 실행
############################
if __name__ == "__main__":
    # =============================
    # 사용자 입력 -> 키워드 추출 후 JSON 생성
    # =============================
    file_user_input = './user_input.xlsx'
    user_input = pd.read_excel(file_user_input)

    ####가중치 계산하는 코드 추가#######
    columns = ['주요업무중요도', '자격요건및우대사항중요도', '혜택및복지중요도']
    row_sums = user_input[columns].sum(axis=1)
    user_input[columns] = user_input[columns].div(row_sums, axis=0)
    user_input.to_excel('./user_input_weighted.xlsx', index=False)
    ####가중치 계산하는 코드 추가#######

    # '시행' 열 기준으로 데이터 필터링
    user_input_filtered = user_input[user_input['시행'].notna()]

    # 키워드 추출 JSON 생성
    user_input_keyword = []

    # ✅ Gemini 호출 횟수 카운트 추가
    api_call_count = 0  

    for idx, row in user_input_filtered.iterrows():
        
        # ✅ 15회마다 60초 대기
        if api_call_count > 0 and api_call_count % 15 == 0:
            print("⏳ 키워드 추출 중 60초 대기 중...")
            time.sleep(60)

        entry = {
            "num": idx + 1,
            "job_task": {
                "keywords": extract_keywords(str(row.get("주요업무", ""))),
                "weight": row.get("주요업무중요도", 1)
            },
            "job_skills": {
                "keywords": extract_keywords(str(row.get("자격요건및우대사항", ""))),
                "weight": row.get("자격요건및우대사항중요도", 1)
            },
            "job_benefits": {
                "keywords": extract_keywords(str(row.get("혜택및복지", ""))),
                "weight": row.get("혜택및복지중요도", 1)
            }
        }
        user_input_keyword.append(entry)

        api_call_count += 1  # ✅ 호출 횟수 증가

    with open('./user_input_keyword.json', 'w', encoding='utf-8') as f:
        json.dump(user_input_keyword, f, ensure_ascii=False, indent=2)
    # =============================
    # 누적공고 DataFrame으로 변환 (BGE/Jina)
    # =============================
    # 예) 누적공고 DataFrame
    df_raw_bge = pd.read_excel("./bge_누적공고.xlsx")
    df_raw_jina = pd.read_excel("./jina_누적공고.xlsx")

    # ✅ '자격요건'과 '우대사항' 열을 합친 '자격요건및우대사항' 생성
    df_raw_bge['자격요건및우대사항'] = df_raw_bge['자격요건'].fillna("") + " " + df_raw_bge['우대사항'].fillna("")
    df_raw_jina['자격요건및우대사항'] = df_raw_jina['자격요건'].fillna("") + " " + df_raw_jina['우대사항'].fillna("")

    # 기존 DataFrame 필터링 부분 수정
    top_df_BGE = df_raw_bge[['시행', 'rank', '공고제목', '주요업무', '자격요건및우대사항', '혜택및복지']].fillna("")
    top_df_Jina = df_raw_jina[['시행', 'rank', '공고제목', '주요업무', '자격요건및우대사항', '혜택및복지']].fillna("")
    
    # =============================
    # Coverage Ratio 계산
    # =============================
    file_keyword = "./user_input_keyword.json"
    with open(file_keyword, "r", encoding="utf-8") as f:
        user_input_keyword = json.load(f)

    start_total_time = time.time()
    api_call_count = 0

    # 각 입력 키워드에 대해 반복
    for idx, user_input_json in enumerate(user_input_keyword, start=1):
        시행번호 = user_input_json.get("num", idx)
        print(f"\n===== [{시행번호}번째 시행] GEMINI EVALUATION + DEBUG =====")
        result_text = evaluate_and_debug_with_gemini(user_input_json, top_df_BGE, top_df_Jina, api_call_count)
        print(result_text)
        
        api_call_count += 1

    end_total_time = time.time()
    total_duration = round(end_total_time - start_total_time, 2)
    print(f"🎯 전체 프로세스 완료! 총 실행 시간: {total_duration}초")

    # =============================
    # 최종 점수 계산
    # =============================
    coverage_df = pd.read_csv("./coverage_results.csv")
    with open('user_input_keyword.json', 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    finscore_df = calculate_finscore(coverage_df, user_data)
    model_scores = finscore_df.groupby('model')['종합점수'].sum()
    finscore_df.to_csv("final_data.csv", encoding="utf-8")

    print("\n📊 모델별 종합점수 합계:")
    print(model_scores.to_string())
    print("\n🏆 결과 비교:")
    if 'bge' in model_scores and 'jina' in model_scores:
        if model_scores['bge'] > model_scores['jina']:
            print(f"✔ bge 모델이 더 우수합니다. (bge: {model_scores['bge']:.4f} > jina: {model_scores['jina']:.4f})")
        elif model_scores['bge'] < model_scores['jina']:
            print(f"✔ jina 모델이 더 우수합니다. (jina: {model_scores['jina']:.4f} > bge: {model_scores['bge']:.4f})")
        else:
            print(f"✔ 두 모델의 점수가 동일합니다. (bge = jina = {model_scores['bge']:.4f})")
    else:
        print("❗ bge 또는 jina 모델이 데이터에 없습니다.")