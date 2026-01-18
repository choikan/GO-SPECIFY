# app.py
# 12



import streamlit as st

from config import GOOGLE_API_KEY, DEBUG, JOBS_CACHE_PATH
from services import fetch_job_list, fetch_job_detail
import ui  # ✅ ui 모듈 전체 import (from ui import ... 연쇄 ImportError 방지)
from utils import load_favorites, calc_dday

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GO - SPECIFY",
    page_icon="🏛",
    layout="wide",
)

ui.apply_global_styles()
st.title("🏛️ GO - SPECIFY")

st.markdown("""
<style>
.subtitle {
    font-size: 1.5rem;
    color: #c7c9cc;
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">공공기관 IT 채용 공고 분석 기반 맞춤형 취업 준비 로드맵 서비스</div>',
    unsafe_allow_html=True
)

# ✅ favorites session init
if "favorites" not in st.session_state:
    st.session_state["favorites"] = load_favorites()

# -----------------------------
# Sidebar Utilities
# -----------------------------
with st.sidebar:
    if st.button("🔄 공고 다시 불러오기"):
        st.session_state.pop("all_jobs_raw", None)
        st.session_state.pop("selected_job", None)
        st.session_state.pop("full_detail", None)
        for k in ["current_keywords", "stage", "selected_certs", "owned_skills"]:
            st.session_state.pop(k, None)
        st.rerun()

    if st.button("🧹 파일 캐시 삭제"):
        try:
            if JOBS_CACHE_PATH.exists():
                JOBS_CACHE_PATH.unlink()
            st.success("캐시 파일 삭제 완료!")
        except Exception as e:
            st.warning(f"캐시 삭제 실패: {e}")

# -----------------------------
# Gemini Configure (optional)
# -----------------------------
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_API_KEY)
        if DEBUG:
            st.sidebar.caption("✅ Gemini configured (DEBUG)")
    except Exception:
        st.warning("⚠️ Gemini 설정에 실패했습니다. AI 기능이 제한될 수 있습니다.")
else:
    st.info("ℹ️ GOOGLE_API_KEY가 없어도 앱은 동작합니다. (AI는 예시 응답으로 표시됩니다.)")

# -----------------------------
# Load Jobs (session)
# -----------------------------
if "all_jobs_raw" not in st.session_state:
    with st.spinner("공고 데이터를 불러오는 중..."):
        st.session_state["all_jobs_raw"] = fetch_job_list(st=st)

filters = ui.render_sidebar(st.session_state["all_jobs_raw"])

# -----------------------------
# Apply Filters (+ search query) + Sort
# -----------------------------
def _job_id(job: dict) -> str:
    return str(job.get("recrutPblntSn") or job.get("pbancId") or job.get("id") or "").strip()


def _get_job_end(job: dict) -> str:
    return str(job.get("pbancEndYmd") or job.get("recrutEndYmd") or "").strip()


def apply_filters(all_jobs, filters):
    filtered = list(all_jobs)

    # ⭐ Favorites only
    if filters.get("favorites_only"):
        favs = st.session_state.get("favorites", set())
        filtered = [j for j in filtered if (_job_id(j) in favs)]

    q = (filters.get("query") or "").strip().lower()
    if q:

        def hay(j):
            return " ".join(
                [
                    str(j.get("recrutPbancTtl", "")),
                    str(j.get("instNm", "")),
                    str(j.get("ncsCdNmLst", "")),
                    str(j.get("workRgnNmLst", "")),
                    str(j.get("recrutSeNm", "")),
                    str(j.get("hireTypeNmLst", "")),
                ]
            ).lower()

        filtered = [j for j in filtered if q in hay(j)]

    if filters.get("institution"):
        filtered = [j for j in filtered if str(j.get("instNm", "")).strip() in filters["institution"]]

    if filters.get("region"):
        filtered = [
            j
            for j in filtered
            if any(reg in (str(j.get("workRgnNmLst", "")) or "") for reg in filters["region"])
        ]

    if filters.get("education") and filters["education"] != "전체":
        filtered = [j for j in filtered if filters["education"] in (str(j.get("acbgCondNmLst", "")) or "")]

    if filters.get("hire_type"):
        filtered = [
            j for j in filtered if any(h in (str(j.get("hireTypeNmLst", "")) or "") for h in filters["hire_type"])
        ]

    if filters.get("career") and filters["career"] != "전체":
        filtered = [j for j in filtered if filters["career"] == (str(j.get("recrutSeNm", "")) or "")]

    # ✅ (B) 정렬 적용
    if filters.get("sort_mode") == "마감순 (D-day)":

        def sort_key(j):
            d = calc_dday(_get_job_end(j))
            # None(날짜없음) -> 맨 아래
            if d is None:
                return 10**9
            # 마감된 공고(d<0) -> 아래쪽으로
            if d < 0:
                return 10**8 + abs(d)
            return d

        filtered.sort(key=sort_key)

    return filtered


jobs = apply_filters(st.session_state["all_jobs_raw"], filters)

# -----------------------------
# Screen Routing
# -----------------------------
if st.session_state.get("selected_job") is not None:
    if st.button("⬅ 목록으로 돌아가기"):
        st.session_state["selected_job"] = None
        st.session_state["full_detail"] = None
        for k in ["current_keywords", "stage", "selected_certs", "owned_skills"]:
            st.session_state.pop(k, None)
        st.rerun()

    if st.session_state.get("full_detail") is None:
        with st.spinner("상세 공고 및 첨부파일 로드 중..."):
            st.session_state["full_detail"] = fetch_job_detail(st.session_state["selected_job"], st=st)

    detail = st.session_state.get("full_detail")
    if detail:
        ui.render_job_detail(detail)
        ui.render_gap_analysis(detail)
else:
    st.info(f"현재 {len(jobs)}개의 공고가 필터링되었습니다.")
    ui.render_job_list(jobs)
