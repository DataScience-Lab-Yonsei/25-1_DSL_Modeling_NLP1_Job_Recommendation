import threading
import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def crawl_and_save(url, output_path):
    # -----------------------
    # 1. 드라이버 세팅
    # -----------------------
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # -----------------------
    # 2. 사이트 접속
    # -----------------------
    driver.get(url)
    time.sleep(2)  # 초기에 페이지 로딩을 위해 잠시 대기

    # -----------------------
    # 3. 스크롤 내리기
    # 필요 시 횟수 늘려서 크롤링 범위 확장
    # -----------------------
    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(50):  # 원하는 만큼 조절
        body.send_keys(Keys.END)
        time.sleep(1)  # 스크롤 로딩 기다리는 시간(필요시 조절)

    # -----------------------
    # 4. 전체 HTML 코드 가져오기
    # -----------------------
    html = driver.page_source

    # -----------------------
    # 5. BeautifulSoup 파싱
    # -----------------------
    soup = BeautifulSoup(html, "html.parser")

    # -----------------------
    # 6. 공고 목록 찾기
    #    보통 공고 LI는 <li class="Card_Card__aaatv"> 형태
    # -----------------------
    job_list = soup.find_all("li", class_="Card_Card__aaatv")
    jobs = []

    # -----------------------
    # 7. 각 공고 정보 추출
    # -----------------------
    for li in job_list:
        a_tag = li.find("a", attrs={"data-attribute-id": "position__click"})
        if not a_tag:
            continue

        # (1) 공고 제목
        position_name = a_tag.get("data-position-name", "").strip()
        # (2) 회사명
        company_name = a_tag.get("data-company-name", "").strip()
        # (3) 공고 ID
        position_id = a_tag.get("data-position-id", "").strip()
        # (4) 공고 상세 URL
        href = a_tag.get("href", "")
        full_url = "https://www.wanted.co.kr" + href  # 절대 경로

        # (5) 근무 위치 및 경력
        location_tag = li.select_one("span.CompanyNameWithLocationPeriod_CompanyNameWithLocationPeriod__location__4_w0l")
        location_career = location_tag.get_text(strip=True) if location_tag else ""

        # --- '·' 기준으로 분리 ---
        if "·" in location_career:
            split_text = location_career.split("·", maxsplit=1)
            work_location = split_text[0].strip()   # 예) "서울 구로구"
            career_info   = split_text[1].strip()   # 예) "경력 1-3년"
        else:
            work_location = location_career
            career_info = ""

        job_info = {
            "공고 제목": position_name,
            "회사명": company_name,
            "공고 ID": position_id,
            "공고 상세 URL": full_url,
            "근무 위치": work_location,
            "경력": career_info
        }
        jobs.append(job_info)

    # -----------------------
    # 8. 1차 DataFrame 생성
    # -----------------------
    df = pd.DataFrame(jobs, columns=[
        "공고 제목", "회사명", "공고 ID", "공고 상세 URL", "근무 위치", "경력"
    ])

    # ------------------------------------------------------------------------------
    # 여기서부터: 각 공고 상세 URL에 접속하여 '주요업무', '자격요건', '우대사항', '혜택 및 복지' 추가 크롤링
    # ------------------------------------------------------------------------------
    # DataFrame에 컬럼 미리 추가
    df["주요업무"] = ""
    df["자격요건"] = ""
    df["우대사항"] = ""
    df["혜택 및 복지"] = ""

    # 각 공고 상세 페이지에 접속하여 정보 수집
    for i in range(len(df)):
        detail_url = df.loc[i, "공고 상세 URL"]
        driver.get(detail_url)
        time.sleep(1)  # 페이지 로딩 대기
        
        # '상세 정보 더 보기' 버튼이 있으면 클릭 후 재로딩
        # 버튼은 보통 text='상세 정보 더 보기' 이거나 해당 XPath를 통해 찾을 수 있음
        try:
            more_info_button = driver.find_element(By.XPATH, '//button[.//span[text()="상세 정보 더 보기"]]')
            more_info_button.click()
            time.sleep(0.5)
        except:
            pass  # 버튼이 없거나 클릭 실패하면 넘어감
        
        # 상세 페이지 소스 파싱
        detail_html = driver.page_source
        detail_soup = BeautifulSoup(detail_html, "html.parser")

        # 주요 4개 섹션 추출
        sections = detail_soup.find_all("div", class_="JobDescription_JobDescription__paragraph__87w8I")
        
        major_work = ""
        qualification = ""
        preferred = ""
        benefits = ""
        
        for sec in sections:
            h3_elem = sec.find("h3", class_="wds-1y0suvb")
            if not h3_elem:
                continue
            title_text = h3_elem.get_text(strip=True)
            
            content_elem = sec.find("span", class_="wds-wcfcu3")
            content_text = content_elem.get_text("\n", strip=True) if content_elem else ""
            
            # 필요한 항목에 매핑
            if title_text == "주요업무":
                major_work = content_text
            elif title_text == "자격요건":
                qualification = content_text
            elif title_text == "우대사항":
                preferred = content_text
            elif title_text == "혜택 및 복지":
                benefits = content_text

        # 해당 공고의 DataFrame 행에 반영
        df.loc[i, "주요업무"] = major_work
        df.loc[i, "자격요건"] = qualification
        df.loc[i, "우대사항"] = preferred
        df.loc[i, "혜택 및 복지"] = benefits

    # -----------------------
    # 10. 사용 완료 후 드라이버 종료
    # -----------------------
    driver.quit()

    # -----------------------
    # 엑셀 파일 한국어로 보기 위해서 불법문자 제거 후 openpyxl로 저장
    # -----------------------
    def remove_illegal_chars(text):
        if isinstance(text, str):
            return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
        return text

    df = df.applymap(remove_illegal_chars)  # 모든 셀에 대해 적용
    df.to_excel(output_path, index=False, engine="openpyxl")


# ============ 메인 시작 ============ #
if __name__ == "__main__":
    """
    아래 세 개의 (url, 저장 파일 경로)를 각각 별도의 스레드로 실행.
    """

    # 1) 개발
    url_dev = "https://www.wanted.co.kr/wdlist/518?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    output_dev = "./데이터추출_개발전체.xlsx"

    # 2) 경영·비즈니스
    url_biz = "https://www.wanted.co.kr/wdlist/507?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    output_biz = "./데이터추출_경영·비즈니스전체.xlsx"

    # 3) 마케팅·광고
    url_mkt = "https://www.wanted.co.kr/wdlist/523?country=kr&job_sort=job.latest_order&years=-1&locations=all"
    output_mkt = "./데이터추출_마케팅·광고전체.xlsx"

    # 작업 목록 (url, 저장경로) 튜플
    tasks = [
        (url_dev, output_dev),
        (url_biz, output_biz),
        (url_mkt, output_mkt)
    ]

    # 스레드 리스트
    threads = []

    # 각 작업을 스레드로 실행
    for url, path in tasks:
        t = threading.Thread(target=crawl_and_save, args=(url, path))
        t.start()
        threads.append(t)

    # 모든 스레드 종료까지 대기
    for t in threads:
        t.join()

    print("모든 분야 크롤링 및 파일 저장이 완료되었습니다.")
