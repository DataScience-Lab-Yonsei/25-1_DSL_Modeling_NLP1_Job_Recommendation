import re
import pandas as pd
import os

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


df1 = pd.read_excel("/Users/chu_chutrain_/Desktop/데이터추출_개발전체.xlsx")
df2 = pd.read_excel("/Users/chu_chutrain_/Desktop/데이터추출_경영·비즈니스전체.xlsx")
df3 = pd.read_excel("/Users/chu_chutrain_/Desktop/데이터추출_마케팅·광고전체.xlsx")

# 기준이 되는 첫 번째 데이터프레임의 열 정보 저장
columns_standard = list(df1.columns)

# 모든 데이터프레임의 열을 비교 
dfs = [df2, df3]
all_same = all(list(df.columns) == columns_standard for df in dfs)

# 결과 출력
if all_same:
    print("모든 데이터프레임의 열이 동일합니다.")
else:
    print("데이터프레임의 열이 서로 다릅니다.")

# 만약 차이가 있다면, 각 데이터프레임의 열 이름을 출력
for i, df in enumerate([df1, df2, df3], start=1):
    print(f"df{i} 열 목록: {list(df.columns)}")

# 행기준 df 병합
df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# 열 이름 변경
df.rename(columns={
    '공고 제목': '공고제목',
    '공고 ID': '공고id',
    '공고 상세 URL': '공고상세url',
    '혜택 및 복지': '혜택및복지',
    '근무 위치': '근무위치'
}, inplace=True)

# 주요업무, 자격요건, 우대사항, 혜택및복지 열이 모두 결측값인 행 삭제
df = df.dropna(subset=['주요업무', '자격요건', '우대사항', '혜택및복지'], how='all')
# 자격요건, 우대사항 열이 모두 결측값인 행 삭제
df = df.dropna(subset=['자격요건', '우대사항'], how='all')

# "세종"인 경우는 유지하고, 그 외 공백이 없는 값들은 삭제
df = df[(df['근무위치'] == "세종") | df['근무위치'].str.contains(r'\s', na=False)]

def is_valid_location(location):
    """
    주어진 근무위치(location)가 올바른지 확인하는 함수.
    """
    # "세종"인 경우 예외적으로 유효한 값으로 인정
    if location.strip() == "세종":
        return True

    # 공백을 기준으로 지역을 분리
    parts = location.split()
    
    # 지역이 두 개 단어로 되어 있어야 함 (예: '서울 강남구')
    if len(parts) != 2:
        return False
    
    region, subregion = parts  # 예: '서울', '강남구'
    
    # region이 location_dict에 존재하고, 해당 region의 리스트 내에 subregion이 존재하는지 확인
    return region in location_dict and subregion in location_dict[region]

# df에서 잘못된 근무위치가 있는 행을 제거
df = df[df['근무위치'].apply(is_valid_location)]

# "경력" 전처리 함수
def clean_experience(text):
    if pd.isna(text):
        return text  # NaN 값 유지
    
    text = text.strip()  # 공백 제거
    
    if "신입" in text:  # "신입"이면 0으로 변환
        return 0
    
    # 숫자 범위 처리 (예: "5-15년", "3~7년")
    match = re.search(r"(\d+)[\-\~](\d+)", text)
    if match:
        return int(match.group(1))  # 최소 숫자 반환
    
    # 최소 연차만 있는 경우 (예: "8년 이상", "5년 경력")
    match = re.search(r"(\d+)년?", text)
    if match:
        return int(match.group(1))
    
    return text  # 변환할 수 없는 경우 원래 값을 유지

# "경력" 컬럼 전처리 적용
df["경력"] = df["경력"].astype(str).apply(clean_experience)

######### Streamlit에 사용할 df
df.to_excel("./all_raw.xlsx",index=False)


######### 여기서부터 임베딩에 사용할 전처리
# 특수문자 리스트 정의 ('ㆍ' 추가됨)
special_chars_pattern = r'[!"#$%&\'()*+,\-./:<=>?\[\]`|~·​‌–—‘’“”•‣…※∙⋄⎮⎸┕■□▲▶▷◈◎●✅⦁⩗「」『』【】・\ufeff＊＜＞｜🍫🎯👉📌📖📚ㆍ⊙③②①ㄴº◆►︎︎︎ㅣ―]'

# "공고 제목" 컬럼 전처리 함수
def clean_title(text):
    if pd.isna(text):
        return text
    text = re.sub(r"\(.*?\)", "", text)  # 소괄호 안 내용 삭제
    text = re.sub(r"\[.*?\]", "", text)  # 대괄호 안 내용 삭제
    text = re.sub(special_chars_pattern, "", text)  # 모든 특수문자 삭제
    text = re.sub(r"\b\d+\s+", "", text)  # "숫자 + 공백" 패턴 삭제
    text = re.sub(r"\s+", " ", text)  # 다중 공백 제거
    return text.strip()

# "주요업무", "자격요건", "혜택 및 복지", + "우대사항" 컬럼 전처리 함수
def clean_text(text):
    if pd.isna(text):
        return text
    text = re.sub(r"(\d+)[^\w\sㄱ-ㅎ가-힣]+(\d+)", r"\1", text)  # "숫자+특수기호+숫자" 패턴 → 첫 번째 숫자만 유지
    text = re.sub(r"\[.*?\]", "", text)  # 대괄호 안 내용 삭제
    text = re.sub(r"\{.*?\}", "", text)  # 중괄호 안 내용 삭제
    text = re.sub(r"<.*?>", "", text)  # 꺽쇠괄호 안 내용 삭제
    text = re.sub(r"\(", "", text)  # 여는 소괄호 삭제
    text = re.sub(r"\)", "", text)  # 닫는 소괄호 삭제
    text = re.sub(special_chars_pattern, "", text)  # 모든 특수기호 삭제
    text = re.sub(r"\b\d+\s+", "", text)  # "숫자 + 공백" 패턴 삭제
    text = re.sub(r"\b\d+[^\w\sㄱ-ㅎ가-힣]+\b", "", text)  # "숫자+특수기호" 패턴 삭제
    text = re.sub(r"\s+", " ", text)  # 다중 공백 제거
    return text.strip()

# 컬럼별 전처리 적용
df["공고제목"] = df["공고제목"].astype(str).apply(clean_title)
df["주요업무"] = df["주요업무"].astype(str).apply(clean_text)
df["자격요건"] = df["자격요건"].astype(str).apply(clean_text)
df["우대사항"] = df["우대사항"].astype(str).apply(clean_text)
df["혜택및복지"] = df["혜택및복지"].astype(str).apply(clean_text)

df["자격요건및우대사항"] = df["자격요건"].fillna("") + " " + df["우대사항"].fillna("")

df.to_excel("./all.xlsx",index=False)




