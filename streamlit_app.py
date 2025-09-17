# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공식 공개 데이터(먼저 시도): NOAA / NASA 계열의 'Global Sea Level' 시계열(월별)을 자동으로 불러오려 시도합니다.
  (출처 URL: https://psl.noaa.gov/data/timeseries/month/SEALEVEL/, https://sealevel.nasa.gov/)
  만약 API/다운로드 실패 시, 예시/대체 데이터를 사용하며 화면에 한국어 안내를 표시합니다.
- 사용자 입력 데이터: /mnt/data/속초해수욕장_전체 다운로드 (1).csv (앱 실행 중 파일 업로드 요구하지 않음)
- 한국어 UI, Pretendard 폰트 적용 시도 (없으면 자동 생략)
- 규칙: 날짜(date), 값(value), group(optional) 표준화, 결측/중복/미래 데이터 제거, CSV 다운로드 버튼 제공.
"""

import io
import os
import sys
from datetime import datetime, timezone, timedelta
import pytz
import re

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# ------------------------------
# 설정: 경로 및 상수
# ------------------------------
# 공식 데이터(우선 시도할 URL들)
OFFICIAL_URLS = [
    "https://psl.noaa.gov/data/timeseries/month/SEALEVEL/",  # NOAA PSL page (시도)
    "https://sealevel.nasa.gov/",                            # NASA Sea Level portal (시도)
]

# 사용자 제공 파일 (앱 내 업로드 금지: 개발자가 업로드한 파일을 사용)
USER_CSV_PATH = "/mnt/data/속초해수욕장_전체 다운로드 (1).csv"

# Pretendard 폰트 파일 (요청된 경로). 존재하면 matplotlib에 등록 시도.
REQUESTED_FONT_PATH = "/fonts/Pretendard-Bold.ttf"
# fallback developer-upload path (if available in environment)
DEV_FONT_PATH = "/mnt/data/Pretendard_Bold (1).ttf"

# 지역 타임존 (사용자 로컬 = Asia/Seoul)
LOCAL_TZ = pytz.timezone("Asia/Seoul")

# ------------------------------
# 유틸리티: 폰트 등록 (시도)
# ------------------------------
def try_register_font():
    try:
        from matplotlib import font_manager
        font_path = None
        if os.path.exists(REQUESTED_FONT_PATH):
            font_path = REQUESTED_FONT_PATH
        elif os.path.exists(DEV_FONT_PATH):
            font_path = DEV_FONT_PATH
        if font_path:
            font_manager.fontManager.addfont(font_path)
            # 추정 폰트 이름 'Pretendard' 사용
            plt.rcParams['font.family'] = "Pretendard"
            # plotly에서 사용하도록 설정으로 전달할 수 있음 (layout.font.family)
            return "Pretendard"
    except Exception:
        pass
    return None

FONT_FAMILY = try_register_font()

# ------------------------------
# 캐시된 데이터 로드/전처리
# ------------------------------
@st.cache_data(ttl=60*60)  # 1시간 캐시
def fetch_official_sea_level():
    """
    시도:
      1) OFFICIAL_URLS 목록을 순서대로 요청 -> 성공 시 내용 가공
      2) 실패하면 예시 데이터(합성 시계열)를 반환 (앱에서 사용자에게 한국어로 안내됨)
    반환값: (df, used_url, is_fallback_bool, message)
      df: columns -> ['date', 'value'] (date: datetime.date)
    """
    headers = {"User-Agent": "streamlit-sea-level-dashboard/1.0"}
    for url in OFFICIAL_URLS:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 100:
                text = resp.text
                # 간단 파싱 시도: 페이지 안에 CSV 링크 혹은 숫자 시계열이 있는지 찾아본다.
                # 1) 숫자형 시계열(예: 연도 또는 연-월과 값)이 포함되어 있으면 regex로 추출
                # 2) 아니라면, 다음 URL 시도
                # regex: 연도-월 패턴 또는 연도만 있는 패턴 탐색
                # 이 파서는 다양한 포맷을 견딜 수 있게 설계됨.
                # 예시 패턴: 1993-01    0.0123  또는 1993 0.0123
                pattern = re.compile(r"(\d{4})[^\d\n\r]{1,4}(\d{1,2})[^\d\n\r]{1,6}([-+]?\d+\.\d+)")
                matches = pattern.findall(text)
                rows = []
                if matches:
                    for y, m, v in matches:
                        try:
                            d = datetime(int(y), int(m), 1).date()
                            val = float(v)
                            rows.append((d, val))
                        except Exception:
                            continue
                else:
                    # fallback: 연도 + 값 쌍 찾기
                    pattern2 = re.compile(r"(\d{4})\D+([-+]?\d+\.\d+)")
                    matches2 = pattern2.findall(text)
                    for y, v in matches2:
                        try:
                            d = datetime(int(y), 1, 1).date()
                            val = float(v)
                            rows.append((d, val))
                        except Exception:
                            continue

                if rows:
                    df = pd.DataFrame(rows, columns=["date", "value"])
                    # 정렬/중복제거
                    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
                    return df, url, False, "공식 데이터 불러오기 성공"
                # else: 시도 실패 -> 다른 URL 시도
        except Exception:
            continue

    # 모든 시도 실패 -> 예시 데이터 생성 (월별 합성 시계열)
    rng = pd.date_range(end=datetime.now(LOCAL_TZ).date(), periods=360, freq="M")
    # 합성 신호: 완만한 증가 + 계절성 + 잡음
    t = np.arange(len(rng))
    values = 0.02 * (t / 12) + 0.01 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.002, size=len(t))
    df_example = pd.DataFrame({"date": rng.date, "value": values})
    return df_example, None, True, "공식 데이터 불러오기 실패 — 예시 데이터로 대체됨"

@st.cache_data(ttl=60*60)
def load_user_csv(path):
    """
    사용자 CSV를 읽고 표준화 시도.
    규칙: 표준화된 columns -> 'date', 'value', 'group' (optional)
    """
    if not os.path.exists(path):
        return None, f"지정된 사용자 CSV 파일이 존재하지 않습니다: {path}"
    try:
        # 인코딩 감지 없이 pandas가 자동 인식하도록 시도
        df_raw = pd.read_csv(path)
    except Exception as e:
        try:
            df_raw = pd.read_csv(path, encoding='cp949')
        except Exception as e2:
            return None, f"CSV 로딩 실패: {e} / {e2}"

    df = df_raw.copy()

    # 컬럼명 소문자화/공백제거
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 칼럼 찾기: 'date', '날짜', '일자', 'year', '년' 등
    date_col = None
    for candidate in ['date', '날짜', '일자', 'Date', 'DATE', 'year', 'Year', 'YearMonth', '년', '일']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # 시도: 첫번째 datetime-like 칼럼 찾기
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().sum() > 0:
                    date_col = c
                    break
            except Exception:
                continue
    if date_col is None:
        return None, "CSV에서 날짜 칼럼을 찾을 수 없습니다."

    # 값 칼럼 찾기: numeric 컬럼 중 하나
    value_col = None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 제외: 만약 날짜 칼럼이 numeric(연도) 중 하나면 제외하지 않음
    for c in numeric_cols:
        # skip if c is same as date_col
        if c == date_col:
            continue
        value_col = c
        break
    if value_col is None:
        # try to find columns named 'value','값','count','거리','수','Visitors' 등
        for candidate in ['value','값','count','수','visitors','Visitors','온도','temperature','관측치']:
            if candidate in df.columns:
                value_col = candidate
                break
    if value_col is None:
        # if still none, try to coerce a non-numeric column
        for c in df.columns:
            if c != date_col:
                try:
                    tmp = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
                    if tmp.notna().sum() > 0:
                        df[c] = tmp
                        value_col = c
                        break
                except Exception:
                    continue
    if value_col is None:
        return None, "CSV에서 값(value) 칼럼을 찾을 수 없습니다."

    # 그룹 칼럼 (선택)
    group_col = None
    for candidate in ['group','Group','구분','type','category','Category']:
        if candidate in df.columns:
            group_col = candidate
            break

    # 변환
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['value'] = pd.to_numeric(df[value_col].astype(str).str.replace(',',''), errors='coerce')
    if group_col:
        df['group'] = df[group_col].astype(str)
    else:
        df['group'] = "전체"

    # 전처리: 결측/중복/미래 데이터 제거
    df = df[['date','value','group']]
    df = df.dropna(subset=['date','value']).drop_duplicates().sort_values('date').reset_index(drop=True)

    # 오늘(로컬 자정) 이후 데이터 제거
    now_local = datetime.now(LOCAL_TZ)
    today_local_date = now_local.date()
    df = df[df['date'].dt.date <= today_local_date].reset_index(drop=True)

    return df, "성공"

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Streamlit + Codespaces 데이터 대시보드", layout="wide")

# 상단 제목
st.title("데이터 대시보드 (공식 데이터 → 사용자 데이터 순서로 표시)")

if FONT_FAMILY:
    st.caption(f"UI 폰트(시도 적용): {FONT_FAMILY}")
else:
    st.caption("Pretendard 폰트가 발견되지 않아 기본 폰트로 표시됩니다.")

# ------------------------------
# 공식 공개 데이터 섹션
# ------------------------------
st.header("1) 공식 공개 데이터: 전 지구 평균 해수면(예시/NOAA/NASA 계열 자동 시도)")

with st.spinner("공식 데이터 불러오는 중..."):
    official_df, used_url, is_fallback, official_msg = fetch_official_sea_level()

if is_fallback:
    st.warning("공식 데이터 불러오기에 실패하여 예시(합성) 데이터로 대체되었습니다. (원본 출처 시도: NOAA / NASA)")
    st.info("원본 데이터 및 API 사용을 원하시면 인터넷 연결 및 해당 기관의 데이터 링크로 접근 가능한지 확인하세요.")
else:
    st.success(f"공식 데이터 불러오기 성공: {used_url}")

# 표준화/전처리 (공식)
def preprocess_timeseries(df):
    df2 = df.copy()
    # Ensure 'date' is datetime
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
    df2 = df2.dropna(subset=['date']).drop_duplicates().sort_values('date').reset_index(drop=True)
    # Rename value col if needed
    if 'value' not in df2.columns:
        # try to find numeric column as value
        for c in df2.columns:
            if c != 'date' and np.issubdtype(df2[c].dtype, np.number):
                df2 = df2.rename(columns={c: 'value'})
                break
    # Future data 제거 (오늘 로컬 자정 이후)
    now_local = datetime.now(LOCAL_TZ)
    today_local_date = now_local.date()
    df2 = df2[df2['date'].dt.date <= today_local_date].reset_index(drop=True)
    return df2

official_df = preprocess_timeseries(official_df)

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("공식 데이터 시계열")
    st.write("데이터 요약:")
    st.dataframe(official_df.head(10))

    # 간단 시각화: 꺾은선 + 영역
    fig = px.area(official_df, x='date', y='value', labels={'date':'날짜','value':'값'}, title="공식 데이터: 해수면 시계열(월별)")
    if FONT_FAMILY:
        fig.update_layout(font_family=FONT_FAMILY)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("다운로드 / 메타")
    if used_url:
        st.markdown(f"- 원본 시도 URL: `{used_url}`")
    st.markdown(f"- 데이터 상태: {'예시 데이터 사용' if is_fallback else '공식 데이터 사용'}")
    st.download_button(
        label="전처리된 공식 데이터 CSV로 다운로드",
        data=official_df.to_csv(index=False).encode('utf-8'),
        file_name="official_sea_level_preprocessed.csv",
        mime="text/csv"
    )

# ------------------------------
# 사용자 입력 데이터 섹션
# ------------------------------
st.header("2) 사용자 제공 데이터 대시보드 (앱 실행 시 내장된 CSV 사용)")

user_df, user_msg = load_user_csv(USER_CSV_PATH)

if user_df is None:
    st.error(f"사용자 데이터 로딩 실패: {user_msg}")
    st.info("앱은 실행 중 파일 업로드를 요구하지 않습니다. 올바른 CSV 경로가 존재하는지 확인하세요.")
else:
    st.success("사용자 CSV 로딩 및 표준화 완료")
    st.write(f"- 원본 파일 경로: `{USER_CSV_PATH}`")
    st.write(f"- 로드 결과: 행 {len(user_df):,}개, 그룹 수 {user_df['group'].nunique():,}개")

    # 사이드바 자동 구성: 기간 필터, 스무딩(이동평균), 그룹 선택
    st.sidebar.header("사용자 데이터 필터 / 옵션 (자동 구성)")
    min_date = user_df['date'].min().date()
    max_date = user_df['date'].max().date()
    date_range = st.sidebar.date_input("기간 선택", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    smoothing = st.sidebar.checkbox("이동평균 적용 (7일 또는 3기간)", value=False)
    smoothing_window = st.sidebar.slider("이동평균 윈도우", min_value=2, max_value=30, value=7) if smoothing else None
    selected_groups = st.sidebar.multiselect("그룹 선택 (없으면 전체)", options=sorted(user_df['group'].unique()), default=sorted(user_df['group'].unique()))

    # 필터 적용
    start_date, end_date = date_range
    mask = (user_df['date'].dt.date >= start_date) & (user_df['date'].dt.date <= end_date) & (user_df['group'].isin(selected_groups))
    df_vis = user_df.loc[mask].copy()
    if df_vis.empty:
        st.warning("선택한 필터에 해당하는 데이터가 없습니다.")
    else:
        # 시계열/비율/지역 판별: 간단 heuristic
        is_time_series = True if df_vis['date'].nunique() > 1 else False
        is_ratio = True if (df_vis['value'].max() <= 1 and df_vis['value'].min() >= 0) and (df_vis['value'].dtype == float) else False
        # 지도: 컬럼에 'lat' or 'latitude' and 'lon' or 'longitude'가 있?
        lat_cols = [c for c in user_df.columns if c.lower() in ('lat','latitude','위도')]
        lon_cols = [c for c in user_df.columns if c.lower() in ('lon','longitude','경도')]
        has_geo = len(lat_cols) > 0 and len(lon_cols) > 0

        st.subheader("사용자 데이터 미리보기")
        st.dataframe(df_vis.head(20))

        # 시각화 선택 (자동)
        if has_geo:
            st.subheader("지도 시각화 (자동 선택)")
            latc = lat_cols[0]
            lonc = lon_cols[0]
            # 간단히 st.map으로
            map_df = df_vis[[latc, lonc]].dropna().rename(columns={latc:'lat', lonc:'lon'})
            if not map_df.empty:
                st.map(map_df)
            else:
                st.info("지도에 표시할 좌표 데이터가 충분하지 않습니다.")
        elif is_time_series:
            st.subheader("시계열 시각화 (자동 선택: 꺾은선/영역)")
            # 그룹별 시계열
            if df_vis['group'].nunique() > 1:
                fig2 = px.line(df_vis, x='date', y='value', color='group', labels={'date':'날짜','value':'값','group':'그룹'}, title="사용자 데이터: 그룹별 시계열")
            else:
                fig2 = px.area(df_vis, x='date', y='value', labels={'date':'날짜','value':'값'}, title="사용자 데이터: 시계열")
            if FONT_FAMILY:
                fig2.update_layout(font_family=FONT_FAMILY)
            st.plotly_chart(fig2, use_container_width=True)

            # smoothing
            if smoothing and smoothing_window and smoothing_window > 1:
                df_sm = df_vis.sort_values('date').copy()
                df_sm['value_smooth'] = df_sm.groupby('group')['value'].transform(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean())
                st.subheader(f"이동평균(윈도우={smoothing_window}) 적용 결과")
                if df_sm['group'].nunique() > 1:
                    fig3 = px.line(df_sm, x='date', y='value_smooth', color='group', labels={'date':'날짜','value_smooth':'스무딩 값','group':'그룹'})
                else:
                    fig3 = px.line(df_sm, x='date', y='value_smooth', labels={'date':'날짜','value_smooth':'스무딩 값'})
                if FONT_FAMILY:
                    fig3.update_layout(font_family=FONT_FAMILY)
                st.plotly_chart(fig3, use_container_width=True)

        else:
            # 비율/분포: 원/도넛 혹은 막대
            st.subheader("비율/분포 시각화 (자동 선택)")
            # 그룹별 합계 비중 막대
            agg = df_vis.groupby('group')['value'].sum().reset_index().sort_values('value', ascending=False)
            if len(agg) <= 8:
                fig4 = px.pie(agg, names='group', values='value', title="그룹별 비중")
            else:
                fig4 = px.bar(agg, x='group', y='value', title="그룹별 합계 (상위 그룹)")
            if FONT_FAMILY:
                fig4.update_layout(font_family=FONT_FAMILY)
            st.plotly_chart(fig4, use_container_width=True)

        # 전처리된 표 CSV 다운로드
        csv_bytes = df_vis.to_csv(index=False).encode('utf-8')
        st.download_button("전처리된 사용자 데이터 CSV 다운로드", data=csv_bytes, file_name="user_data_preprocessed.csv", mime="text/csv")

# ------------------------------
# 하단: 도움말 & 주의사항(짧게)
# ------------------------------
st.markdown("---")
st.info("설명: 이 앱은 먼저 공개 '해수면' 관련 시계열 데이터를 자동으로 시도하여 불러옵니다. 실패 시 예시 데이터로 대체됩니다. 이후 앱 실행 시 포함된 사용자 CSV 파일을 자동으로 읽어 대시보드를 생성합니다. (앱 중간에 파일 업로드를 요구하지 않습니다.)")
st.caption("코드 주석에 원본 데이터 시도 URL을 남겼습니다. NOAA/PSL, NASA sealevel 포털 등을 시도합니다.")
