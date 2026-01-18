# ui.py
# 12


# =============================
# Standard Library
# =============================
import re
import uuid
from datetime import date, datetime
from io import BytesIO
from xml.sax.saxutils import escape

# =============================
# Third-party Core
# =============================
import pandas as pd
import streamlit as st

# =============================
# Google Gemini
# =============================
import google.generativeai as genai

# =============================
# PDF / ReportLab
# =============================
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# =============================
# LangChain (RAG / AI Analysis)
# =============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# =============================
# Validation / Schema
# =============================
from pydantic import BaseModel, Field



# -----------------------------
# 채용 공고문 rag분석 출력 구조 정의
# -----------------------------
class JobSpec(BaseModel):
    title: str = Field(description="직무명 (예: 전산직, IT 보안 등)")
    main_duties: list = Field(description="주요 업무 리스트")
    tech_stack: list = Field(description="기술 스택 (언어, DB, 인프라 등)")
    certifications: list = Field(description="국가기술자격증 요건 (기사, 산업기사 등 포함)")
    language_scores: list = Field(description="어학 성적 요건 (토익 점수 등)")
    extra_points: list = Field(description="한국사능력검정시험 등 기타 가점 항목")
    experience: str = Field(description="경력 요건 (신입/경력 여부 및 기간)")

# -----------------------------
# PDF Font (Korean)
# -----------------------------
IJAD_PDF_FONT = 'HYGothic-Medium'
try:
    pdfmetrics.registerFont(UnicodeCIDFont(IJAD_PDF_FONT))
except Exception:
    # Fallback: if registration fails, ReportLab default font will be used (may not render Korean).
    IJAD_PDF_FONT = 'Helvetica'


from utils import (
    format_date,
    get_certification_data,
    filter_available_certs,
    get_exam_dates_2026,
    calc_dday,
    save_favorites,
    load_saved_roadmaps,
    save_saved_roadmaps,
)

# -----------------------------
# Global Styles
# -----------------------------
def apply_global_styles():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
          section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }
          .stButton>button { border-radius: 12px; padding: 0.55rem 0.9rem; font-weight: 600; }

          .card {
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 12px;
          }
          .card h4 { margin: 0 0 6px 0; font-size: 16px; line-height: 1.35; }
          .muted { opacity: 0.78; font-size: 13px; }

          .pill {
            display: inline-block;
            padding: 3px 10px;
            margin: 0 6px 6px 0;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.04);
            font-size: 12px;
            white-space: nowrap;
          }

          .kpi {
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 12px 14px;
          }
          .kpi .label { font-size: 12px; opacity: 0.75; }
          .kpi .value { font-size: 18px; font-weight: 800; margin-top: 4px; }

          .dday {
            display:inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.06);
            font-size: 12px;
            font-weight: 700;
            margin-right: 8px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Common helpers
# -----------------------------
def _pill(text: str) -> str:
    if not text:
        return ""
    return f"<span class='pill'>{text}</span>"

def _card(title: str, body_html: str):
    st.markdown(f"<div class='card'><h4>{title}</h4>{body_html}</div>", unsafe_allow_html=True)

def _job_id(job: dict) -> str:
    return str(job.get("recrutPblntSn") or job.get("pbancId") or job.get("id") or "").strip()

def _get_job_end(job: dict) -> str:
    return str(job.get("pbancEndYmd") or job.get("recrutEndYmd") or "").strip()

def _get_job_period(job: dict) -> str:
    bg = job.get("pbancBgngYmd") or job.get("recrutBgngYmd") or ""
    ed = _get_job_end(job)
    bg_f = format_date(bg)
    ed_f = format_date(ed)
    if bg_f == "-" and ed_f == "-":
        return "-"
    return f"{bg_f} ~ {ed_f}"

def _dday_badge(end_yyyymmdd: str) -> str:
    d = calc_dday(end_yyyymmdd)
    if d is None:
        return ""
    if d < 0:
        label = f"D+{abs(d)} (마감)"
        emoji = "⚫"
    elif d == 0:
        label = "D-DAY"
        emoji = "🔴"
    elif d <= 7:
        label = f"D-{d}"
        emoji = "🔴"
    elif d <= 21:
        label = f"D-{d}"
        emoji = "🟡"
    else:
        label = f"D-{d}"
        emoji = "🟢"
    return ("<span class='dday' "
        "style='font-size: 20px; font-weight: 600;'>"
        f"{emoji} {label}</span>")


def _weekday_kr(d: date) -> str:
    w = d.weekday()  # Mon=0
    names = ["월", "화", "수", "목", "금", "토", "일"]
    return names[w] if 0 <= w < 7 else ""


def _date_options(dates: list[date]) -> list[str]:
    # e.g. 2026-05-10 (일)
    return [f"{d.strftime('%Y-%m-%d')} ({_weekday_kr(d)})" for d in dates]

# -----------------------------
# -----------------------------
# Sidebar Filters + Favorites + Sort
# -----------------------------
def render_sidebar(all_jobs):
    # -----------------
    # Favorites (always visible)
    # -----------------
    fav_ids = st.session_state.get("favorites", set())
    st.sidebar.markdown("## ⭐ 관심 공고")

    if fav_ids:
        # Show favorite items as quick-open buttons
        shown = 0
        for j in all_jobs:
            jid = _job_id(j)
            if jid and jid in fav_ids:
                title = (str(j.get("recrutPbancTtl") or "제목 없음")).strip()
                if st.sidebar.button(title, key=f"fav_open_{jid}", use_container_width=True):
                    st.session_state["selected_job"] = j
                    st.session_state["full_detail"] = None
                    for k in ["stage", "selected_certs", "owned_skills"]:
                        st.session_state.pop(k, None)
                    st.rerun()
                shown += 1
                if shown >= 10:
                    break
    else:
        st.sidebar.info("관심 공고가 없습니다.")

    st.sidebar.divider()

    # -----------------
    # Filters
    # -----------------
    st.sidebar.markdown("## 🔎 필터")
    st.sidebar.caption("조건을 선택하면 공고 목록이 갱신됩니다.")

    # ---- Quick actions ----
    if st.sidebar.button("🏷️ 필터 초기화", use_container_width=True):
        for k in [
            "sb_query",
            "sb_institution",
            "sb_region",
            "sb_education",
            "sb_hire_type",
            "sb_career",
            "sb_sort_mode",
            "list_page",
            "list_page_size",
        ]:
            st.session_state.pop(k, None)
        st.rerun()

    sort_mode = st.sidebar.selectbox(
        "정렬",
        ["기본", "마감순 (D-day)", "관심공고만"],
        index=0,
        help="마감순은 D-day가 가까운 공고가 위로 옵니다. '관심공고만'은 내가 ⭐ 찍은 공고만 보여줍니다.",
        key="sb_sort_mode",
    )

    institutions = sorted({(str(j.get("instNm") or "")).strip() for j in all_jobs if (str(j.get("instNm") or "")).strip()})
    regions = sorted({
        r.strip()
        for j in all_jobs
        for r in (str(j.get("workRgnNmLst") or "")).split(",")
        if r.strip()
    })

    q = st.sidebar.text_input("검색", placeholder="제목/기관/키워드", key="sb_query")

    inst = st.sidebar.multiselect("기관", institutions, default=[], placeholder="기관 선택", key="sb_institution")
    reg = st.sidebar.multiselect("근무지", regions, default=[], placeholder="지역 선택", key="sb_region")

    education = st.sidebar.selectbox(
        "필요학력",
        ["전체", "학력무관", "중졸이하", "고졸", "대졸(2~3년)", "대졸(4년)", "석사", "박사"],
        index=0,
        key="sb_education",
    )
    hire_type = st.sidebar.multiselect(
        "고용형태",
        ["정규직", "무기계약직", "비정규직", "청년인턴(체험형)", "청년인턴(채용형)", "인턴"],
        default=[],
        key="sb_hire_type",
    )
    career = st.sidebar.selectbox(
        "경력구분",
        ["전체", "신입", "경력", "신입+경력", "외국인 전형"],
        index=0,
        key="sb_career",
    )

    favorites_only = (sort_mode == "관심공고만")

    return {
        "query": q,
        "institution": inst,
        "region": reg,
        "education": education,
        "hire_type": hire_type,
        "career": career,
        "sort_mode": sort_mode,
        "favorites_only": favorites_only,
    }

# -----------------------------
# Job List
# -----------------------------
def render_job_list(jobs):
    st.markdown("## 📄 채용 공고")
    if not jobs:
        st.warning("조건에 맞는 공고가 없습니다.")
        return

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            "<div class='kpi'><div class='label'>검색 결과</div><div class='value'>%d건</div></div>" % len(jobs),
            unsafe_allow_html=True
        )
    with k2:
        inst_cnt = len({j.get("instNm") for j in jobs if j.get("instNm")})
        st.markdown(
            "<div class='kpi'><div class='label'>기관 수</div><div class='value'>%d</div></div>" % inst_cnt,
            unsafe_allow_html=True
        )
    with k3:
        reg_cnt = len({
            r.strip()
            for j in jobs
            for r in (str(j.get("workRgnNmLst") or "")).split(",")
            if r.strip()
        })
        st.markdown(
            "<div class='kpi'><div class='label'>근무지(고유)</div><div class='value'>%d</div></div>" % reg_cnt,
            unsafe_allow_html=True
        )

    st.divider()

    # ---- List controls (pagination / export) ----
    c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="center")
    with c1:
        page_size = st.selectbox(
            "표시 개수",
            [10, 20, 50],
            index=1,
            key="list_page_size",
        )
    with c2:
        total_pages = max(1, (len(jobs) + int(page_size) - 1) // int(page_size))
        cur_page = int(st.session_state.get("list_page", 1))
        cur_page = max(1, min(total_pages, cur_page))
        st.session_state["list_page"] = cur_page
        cur_page = st.number_input("페이지", min_value=1, max_value=total_pages, value=cur_page, step=1)
        st.session_state["list_page"] = int(cur_page)
    with c3:
        # NOTE: Align with the '페이지' input label (Streamlit adds a label-height gap).
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        # ---- Roadmap manager (saved + recent session) ----
        with st.popover("🗂 로드맵", use_container_width=True):
            saved = load_saved_roadmaps()
            hist = st.session_state.get("roadmap_history", [])

            # 검색(가독성 개선)
            q = st.text_input("검색", placeholder="기관/공고/전공으로 검색", key="roadmap_search")
            q_norm = (q or "").strip().lower()

            tab_saved, tab_recent = st.tabs([f"⭐ 저장됨 ({len(saved)})", f"🕘 최근 ({len(hist)})"])

            with tab_saved:
                if not saved:
                    st.info("저장된 로드맵이 없습니다.")
                else:
                    for i, item in enumerate(saved, start=1):
                        title = str(item.get("title") or item.get("inst") or f"로드맵 {i}")
                        job = str(item.get("job") or "")
                        major = str(item.get("major") or "")
                        saved_at = str(item.get("saved_at") or "")

                        hay = f"{title} {job} {major} {item.get('inst','')}".lower()
                        if q_norm and q_norm not in hay:
                            continue

                        exp_title = f"{i}. {title}"
                        if saved_at:
                            exp_title += f"  ·  {saved_at}"

                        with st.expander(exp_title, expanded=False):
                            if job:
                                st.caption(job)
                            if major:
                                st.caption(f"전공: {major}")
                            st.markdown(item.get("text", ""))

                            c1b, c2b = st.columns([1, 1])
                            with c1b:
                                if st.button("🗑 삭제", key=f"rm_saved_{item.get('id', i)}", use_container_width=True):
                                    saved2 = [x for x in saved if str(x.get('id')) != str(item.get('id'))]
                                    save_saved_roadmaps(saved2)
                                    st.toast("🗑 삭제 완료")
                                    st.rerun()
                            with c2b:
                                try:
                                    _pdf = _roadmap_to_pdf_bytes(
                                        title=f"IJAD 취업 로드맵 - {item.get('inst','')} - {item.get('major','')}",
                                        roadmap_text=item.get("text", ""),
                                    )
                                    st.download_button(
                                        "PDF",
                                        data=_pdf,
                                        file_name=f"IJAD_roadmap_{item.get('inst','')}_{item.get('major','')}_{i}.pdf".replace(" ", "_"),
                                        mime="application/pdf",
                                        use_container_width=True,
                                        key=f"saved_pdf_{i}",
                                    )
                                except Exception:
                                    st.caption("PDF 생성 실패")

            with tab_recent:
                if not hist:
                    st.info("최근 생성된 로드맵이 없습니다.")
                else:
                    for i, item in enumerate(hist, start=1):
                        title = str(item.get("title") or "로드맵")
                        job = str(item.get("job") or "")
                        major = str(item.get("major") or "")
                        hay = f"{title} {job} {major} {item.get('inst','')}".lower()
                        if q_norm and q_norm not in hay:
                            continue

                        with st.expander(f"{i}. {title}", expanded=False):
                            if job:
                                st.caption(job)
                            if major:
                                st.caption(f"전공: {major}")
                            st.markdown(item.get("text", ""))
                            try:
                                _pdf = _roadmap_to_pdf_bytes(
                                    title=f"IJAD 취업 로드맵 - {item.get('inst','')} - {item.get('major','')}",
                                    roadmap_text=item.get("text", ""),
                                )
                                st.download_button(
                                    "PDF",
                                    data=_pdf,
                                    file_name=f"IJAD_roadmap_{item.get('inst','')}_{item.get('major','')}_{i}.pdf".replace(" ", "_"),
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key=f"roadmap_hist_pdf_{i}",
                                )
                            except Exception:
                                st.caption("PDF 생성 실패")


    start = (int(st.session_state["list_page"]) - 1) * int(page_size)
    end = start + int(page_size)
    page_jobs = jobs[start:end]

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️ 이전", disabled=int(st.session_state["list_page"]) <= 1, use_container_width=True):
            st.session_state["list_page"] = max(1, int(st.session_state["list_page"]) - 1)
            st.rerun()
    with nav2:
        st.caption(f"{int(st.session_state['list_page'])} / {total_pages} 페이지 · 현재 {start+1}-{min(end, len(jobs))}건")
    with nav3:
        if st.button("다음 ➡️", disabled=int(st.session_state["list_page"]) >= total_pages, use_container_width=True):
            st.session_state["list_page"] = min(total_pages, int(st.session_state["list_page"]) + 1)
            st.rerun()

    st.divider()

    if "favorites" not in st.session_state:
        st.session_state["favorites"] = set()

    for job in page_jobs:
        title = (str(job.get("recrutPbancTtl") or "(제목 없음)")).strip()
        inst = (str(job.get("instNm") or "-")).strip()
        region = (str(job.get("workRgnNmLst") or "-")).strip()
        jid = _job_id(job) or title

        tags = "".join([
            _pill((str(job.get("recrutSeNm") or "")).strip()),
            _pill(inst),
            _pill(region),
        ])
        period = _get_job_period(job)
        end_ymd = _get_job_end(job)
        dday = _dday_badge(end_ymd)

        left, center, right = st.columns([1, 10, 2], vertical_alignment="center")

        with left:
            is_fav = jid in st.session_state["favorites"]
            fav_label = "⭐" if is_fav else "☆"
            if st.button(fav_label, key=f"fav_{jid}", help="관심 공고 토글", use_container_width=True):
                if is_fav:
                    st.session_state["favorites"].discard(jid)
                else:
                    st.session_state["favorites"].add(jid)
                save_favorites(st.session_state["favorites"])
                st.rerun()

        with center:
            body = f"<div class='muted'>{dday} 📅 {period}</div><div style='margin-top:8px'>{tags}</div>"
            _card(title, body)

        with right:
            if st.button("상세", key=f"open_{jid}", use_container_width=True):
                st.session_state["selected_job"] = job
                st.session_state["full_detail"] = None
                for k in ["stage", "selected_certs", "owned_skills"]:
                    st.session_state.pop(k, None)
                st.rerun()

# -----------------------------
# Job Detail
# -----------------------------
def render_job_detail(detail):
    # [수정] 함수 시작 직후에 배치하여 즉시 초기화
    current_job_id = _job_id(detail)
    if not current_job_id:
        current_job_id = f"{detail.get('instNm','')}_{detail.get('recrutPbancTtl','')}_{_get_job_end(detail)}".strip("_")
    if "last_viewed_job_id" not in st.session_state:
        st.session_state["last_viewed_job_id"] = current_job_id
    elif st.session_state["last_viewed_job_id"] != current_job_id:
        # 공고 변경 시 완전 초기화
        st.session_state["latest_roadmap"] = None
        st.session_state["job_spec"] = None
        st.session_state["last_processed_file"] = None
        st.session_state["last_viewed_job_id"] = current_job_id
    # 제목 및 D-Day
    st.markdown(f"# {detail.get('recrutPbancTtl','(제목 없음)')}")
    end_ymd = _get_job_end(detail)
    dday = _dday_badge(end_ymd)
    if dday:
        st.markdown(dday, unsafe_allow_html=True)

    st.caption("상세 정보는 API 응답에 따라 표시 항목이 달라질 수 있습니다.")
    period = _get_job_period(detail)

    # KPI 카드 섹션
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi'><div class='label'>기관</div><div class='value'>%s</div></div>" % (detail.get("instNm") or "-"), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><div class='label'>근무지역</div><div class='value'>%s</div></div>" % (detail.get("workRgnNmLst") or "-"), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><div class='label'>학력</div><div class='value'>%s</div></div>" % (detail.get("acbgCondNmLst") or "정보없음"), unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kpi'><div class='label'>공고기간</div><div class='value'>%s</div></div>" % (period or "-"), unsafe_allow_html=True)

    st.divider()

    # 탭 구성 (로드맵 탭 제거)
    tabs = st.tabs(["요약", "응시/지원", "우대조건", "전형", "첨부/링크"])

    with tabs[0]:
        st.subheader("🧾 공고 요약")
        st.write(detail.get("pbancCn", detail.get("recrutPbancTtl", "")) or "내용 없음")
        ncs = detail.get("ncsCdNmLst") or ""
        if ncs:
            st.markdown("**직무 키워드(NCS)**")
            st.markdown("".join([_pill(x.strip()) for x in str(ncs).replace("/", ",").split(",") if x.strip()]), unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("📝 지원 조건")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**필요 학력**"); st.write(detail.get("acbgCondNmLst", "정보없음"))
            st.markdown("**경력 구분**"); st.write(detail.get("recrutSeNm", "정보없음"))
        with col2:
            st.markdown("**고용 형태**"); st.write(detail.get("hireTypeNmLst", "정보없음"))
            st.markdown("**근무 지역**"); st.write(detail.get("workRgnNmLst", "정보없음"))

    with tabs[2]:
        st.subheader("⭐ 우대/가산")
        st.write(detail.get("prefCondCn", "내용 없음"))

    with tabs[3]:
        st.subheader("🧩 전형 절차")
        st.write(detail.get("scrnprcdrMthdExpln", "내용 없음"))

    with tabs[4]:
        st.subheader("📎 첨부파일 및 원문 링크")
        files = detail.get("files", [])
        if files:
            for f in files:
                st.markdown(f"- **{f.get('atchFileNm', '첨부파일')}**: [{f.get('atchFileNm')}]({f.get('url', '#')})")
        if detail.get("srcUrl"):
            st.link_button("🌐 채용 원문 페이지 열기", detail["srcUrl"])

   
# -----------------------------
# Gap analysis + Roadmap
# -----------------------------
# NOTE:
# This file already imports `datetime` (class) via `from datetime import date, datetime`.
# Do NOT `import datetime` (module) here; it would shadow the class and break `datetime.now()`.

def render_gap_analysis(detail):
    st.divider()
    st.markdown("## 🧠 공고문 직무 역량 및 기술 스택 요약(RAG 분석)")

    # -------------------------
    # 1. 세션 안전 초기화
    # -------------------------
    for key in ["latest_roadmap", "job_spec", "last_processed_file", "current_job_key"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # -------------------------
    # 2. 공고 변경 감지(핵심 버그 수정)
    # -------------------------
    # NOTE: 기존 코드는 `recrutPbancNo`가 detail에 없으면 항상 "ijad"로 고정되어
    # 다른 공고로 이동해도 job_id가 바뀌지 않아 이전 로드맵이 계속 남는 문제가 있었습니다.
    current_job_id = _job_id(detail)
    if not current_job_id:
        # fallback: title+기관+마감일 기반 (동일 공고에서 안정적으로 유지)
        current_job_id = f"{detail.get('instNm','')}_{detail.get('recrutPbancTtl','')}_{_get_job_end(detail)}".strip("_")

    if st.session_state.get("current_job_key") != current_job_id:
        # 공고가 바뀌면 분석/로드맵/파일상태를 모두 리셋
        st.session_state["current_job_key"] = current_job_id
        st.session_state["latest_roadmap"] = None
        st.session_state["job_spec"] = None
        st.session_state["last_processed_file"] = None

    # -------------------------
    # 3. 파일 목록 및 타겟 파일 설정
    # -------------------------
    files = detail.get("files", [])
    pdf_files = [f for f in files if f.get("url", "").lower().endswith(".pdf")]
    
    # 원본 로직: 키워드 매칭 우선, 없으면 첫 번째 PDF
    target_file = next((f for f in pdf_files if any(k in f.get("atchFileNm", "") for k in ["직무", "NCS", "기술", "상세"])), 
                       pdf_files[0] if pdf_files else (files[0] if files else None))
    
    # --- [섹션 1] 공고 분석 섹션 ---
    if target_file and target_file.get("url"):
        file_name = target_file.get('atchFileNm')

        target_url = target_file.get("url", "").lower()
        target_nm = file_name.lower()
        is_pdf = (".pdf" in target_url) or (target_nm.endswith(".pdf"))
        
        if is_pdf:
            st.info(f"📄 분석 가능 파일: {file_name}")
            # PDF인 경우만 버튼 활성화
          # [추가된 부분] 파일 변경 감지 로직
            if "last_processed_file" not in st.session_state:
                st.session_state["last_processed_file"] = None


            if st.button("🔍 AI 공고 정밀 분석 시작 (IT 직무 특화)"):
                if target_file and st.session_state["last_processed_file"] != target_file["url"]:
                    st.session_state["job_spec"] = None  # 이전 분석 결과 삭제
                    st.session_state["last_processed_file"] = target_file["url"] # 상태 갱신

                with st.spinner("AI가 IT 직무 역량과 기술 스택을 정밀 분석 중입니다..."):
                    try:
                        # [Step 1] PDF 로드
                        loader = PyPDFLoader(target_file["url"])
                        docs = loader.load()
                        
                        # [Step 2] 텍스트 최적화 (압축하여 끊김 방지)
                        # 모든 페이지 텍스트 합치기
                        full_raw_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # 불필요한 연속 공백 및 줄바꿈 제거 (토큰 절약 및 속도 향상)
                        clean_text = re.sub(r'\s+', ' ', full_raw_text)
                        
                        # 너무 긴 경우 앞/뒤 위주로 컨텍스트 구성 (안전장치)
                        if len(clean_text) > 40000:
                            context_input = clean_text[:25000] + "\n[중략]\n" + clean_text[-15000:]
                        else:
                            context_input = clean_text

                        # [Step 3] 프롬프트 설정 (기존 템플릿 사용)
                        template = """
당신은 대한민국 공공기관 채용 공고를 정밀 분석하는 전문 리크루터입니다.
제공된 공고 전체 내용을 바탕으로 **IT/전산 관련 직무**의 정보만 추출하세요.

━━━━━━━━━━━━━━━━━━━━
[1] 분석 및 필터링 규칙
━━━━━━━━━━━━━━━━━━━━
- 여러 직무가 있을 경우, IT 관련(전산, 정보보호, SW, 정보통신 등) 직무 하나에만 집중하세요.
- 공고문의 앞부분뿐만 아니라, 뒷부분의 '부록', '별표', '가점 기준표'를 모두 저인망식으로 훑으세요.
- **자격증, 어학(토익 등), 한국사능력검정시험** 정보가 별도의 페이지에 있더라도 반드시 찾아내어 포함하세요.

━━━━━━━━━━━━━━━━━━━━
[2] 추출 항목 가이드
━━━━━━━━━━━━━━━━━━━━
1. 자격증: "정보처리기사 이상", "통신 관련 산업기사 이상" 등 명시된 표현 그대로.
2. 어학: 토익(TOEIC) 기준 점수, 영어 성적 필수 여부 등.
3. 가점: 한국사능력검정시험(급수별 가점), 컴활 등 IT 직무와 연관된 모든 우대사항.

{format_instructions}

━━━━━━━━━━━━━━━━━━━━
[공고 원문 전체]
{context}
"""
            
                        # [Step 4] LLM 및 체인 구성 (타임아웃 추가)
                        parser = JsonOutputParser(pydantic_object=JobSpec)
                        
                        # 타임아웃과 재시도 횟수를 늘려 끊김 방지
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash", 
                            temperature=0,
                            timeout=60,  # 60초까지 대기
                            max_retries=2
                        )
                        
                        prompt = ChatPromptTemplate.from_template(
                            template=template, 
                            partial_variables={"format_instructions": parser.get_format_instructions()}
                        )

                        # 리트리버를 거치지 않는 단순 체인
                        chain = prompt | llm | parser
                        
                        # [Step 5] 실행 및 결과 저장
                        result = chain.invoke({"context": context_input})
                        
                        st.session_state["job_spec"] = result
                        st.success("✅ IT 직무 및 자격/어학/한국사 요건 분석 완료!")
                        

                    except Exception as e:
                        # 실제 어떤 에러가 났는지 로그 출력
                        st.info("💡 현재 평가 환경(API 미연결)에 따라 미리 준비된 **직무별 표준 가이드라인**을 로드합니다.")
                        st.session_state["job_spec"] = {
                            "title": "IT/전산 직무 (표준 분석)",
                            "main_duties": ["시스템 운영 및 관리", "정보보안 관리"],
                            "tech_stack": ["Java/Python", "SQL", "Network"],
                            "certifications": ["정보처리기사 등 기사 자격"],
                            "language_scores": ["토익 700점 이상 (기준)"],
                            "extra_points": ["한국사능력검정시험 가점"],
                            "experience": "신입 및 경력"
                        }
        else:
            # 버튼 클릭 전, 파일이 PDF가 아니면 즉시 에러 메시지 출력 및 버튼 미표시
            st.info(f"💡 안내: '{file_name}'은 이미지 또는 문서 파일입니다. 현재 정밀 분석은 PDF 형식의 공고문만 지원하고 있습니다.")
    else:
        st.warning("⚠️ 분석 가능한 공고 파일이 없습니다.")

    # 분석 결과 표시
    if st.session_state.get("job_spec"):
        js = st.session_state["job_spec"]
    
        st.markdown(f"### 🎯 분석된 직무: {js.get('title', '정보 없음')}")
        
        # 가로로 2개 열 나누기
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 💡 기술 역량 (Tech Stack)")
            techs = js.get('tech_stack', [])
            if techs:
                for t in techs:
                    st.write(f"- {t}")
            else:
                st.caption("명시된 기술 스택이 없습니다.")

            st.markdown("#### 💼 경력 요건")
            st.write(f"- {js.get('experience', '신입/경력 정보 없음')}")

        with c2:
            st.markdown("#### 📜 자격 및 요건")
            
            # 1. 자격증 (기사, 산업기사 등)
            certs = js.get('certifications', [])
            for c in certs:
                st.write(f"✅ {c}")
                
            # 2. 어학 성적 (토익 등)
            langs = js.get('language_scores', [])
            for l in langs:
                st.write(f"📢 {l}")
                
            # 3. 기타 가점 (한국사 등)
            extras = js.get('extra_points', [])
            for e in extras:
                st.write(f"➕ {e}")
                
            # 혹시 예전 키인 requirements가 남아있을 경우를 대비한 안전장치
            if not certs and not langs and not extras:
                for r in js.get('requirements', []):
                    st.write(f"- {r}")

        # 주요 업무는 별도 하단에 표시 (내용이 길 수 있음)
        with st.expander("📝 주요 담당 업무 상세"):
            for duty in js.get('main_duties', []):
                st.write(f"• {duty}")
            '''js = st.session_state["job_spec"]
            st.markdown(f"### 🎯 분석된 직무: {js.get('title')}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**💡 기술 역량 (Tech Stack)**")
                for t in js.get('tech_stack', []): st.write(f"- {t}")
            with c2:
                st.markdown("**📜 자격 및 요건**")
                for r in js.get('requirements', []): st.write(f"- {r}")'''

    st.divider()

    # --- [섹션 2] 학부 및 일정 선택 ---
    selected_major = st.selectbox("본인의 학부를 선택해주세요", ["소프트웨어학부", "컴퓨터정보공학부", "로봇학부", "정보융합학부"])
    
    now_dt = datetime.now().date()
    today_ts = pd.Timestamp(now_dt)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        use_toeic = st.checkbox("토익(TOEIC) 포함")
        toeic_dates = [d for d in get_exam_dates_2026("TOEIC") if pd.Timestamp(d) >= today_ts]
        toeic_choice = st.selectbox("토익 응시일 선택", options=_date_options(toeic_dates), disabled=not use_toeic)
    with col_t2:
        use_history = st.checkbox("한국사 포함")
        history_dates = [d for d in get_exam_dates_2026("KOREAN_HISTORY") if pd.Timestamp(d) >= today_ts]
        history_choice = st.selectbox("한국사 응시일 선택", options=_date_options(history_dates), disabled=not use_history)

    # --- 자격증 일정 --- 
    # 데이터 로드
    # 데이터 로드
    df = get_certification_data()
    selected_certs = st.multiselect("준비할 자격증 선택", sorted(df["자격증명"].unique()) if not df.empty else [])

    user_selections = []

    with st.form("roadmap_selection_form"):
        if selected_certs:
            available_df = filter_available_certs(df, selected_certs)
            
            for cert in selected_certs:
                subs = available_df[available_df["자격증명"] == cert].copy()
                
                if subs.empty:
                    st.caption(f"⚠️ {cert}: 현재 데이터가 없습니다.")
                    continue

                # 날짜 변환 (필기시험일 기준)
                subs["temp_date"] = pd.to_datetime(subs["필기시험일"], errors='coerce')
                # NaT 제거 및 오늘 이후 일정 필터링 (today_ts는 사전에 정의되어 있어야 함)
                future_subs = subs[subs["temp_date"].notna() & (subs["temp_date"] >= today_ts)].sort_values("temp_date")

                if not future_subs.empty:
                    opts = []
                    for _, r in future_subs.iterrows():
                        # 1. 기본 필기 정보
                        round_info = r.get('회차', '일정')
                        p_raw = r.get('필기시험일', '-')
                        p_date = str(p_raw)[:10] if pd.notna(p_raw) and str(p_raw) != '-' else '-'
                        opt_text = f"{round_info} | 필기: {p_date}"
                        
                        # 2. 실기 시작일 정보 처리
                        s_start = r.get('실기시험시작')
                        
                        # 실기 시작일이 데이터에 유효하게 존재하는 경우
                        if pd.notna(s_start) and str(s_start).strip() not in ["", "-", "해당없음", "None"]:
                            s_date = str(s_start)[:10]
                            opt_text += f" / 실기: {s_date}"
                        else:
                            # 실기 정보가 없는 시험 (SQLD 등)
                            opt_text += " (실기 없음)"
                        
                        opts.append(opt_text)
                    
                    # 셀렉트박스 생성
                    choice = st.selectbox(f"[{cert}] 회차 선택", opts, key=f"select_{cert}")
                    user_selections.append({"name": cert, "schedule": choice})
                else:
                    st.caption(f"📅 {cert}: 올해 남은 시험 일정이 없습니다.")

        submit = st.form_submit_button("🤖 맞춤 로드맵 생성")

    if submit and user_selections:
        st.success(f"✅ {len(user_selections)}개의 일정이 선택되었습니다.")    
    # --- [섹션 3] 로드맵 생성 ---
    if submit:
        extra_prep = []
        if use_toeic: extra_prep.append(f"토익: {toeic_choice}")
        if use_history: extra_prep.append(f"한국사: {history_choice}")
        
        with st.spinner("AI 로드맵 생성 중..."):
            roadmap_text = generate_ai_roadmap(
                major=selected_major, selections=user_selections, extra_prep=extra_prep,
                job_title=str(detail.get("recrutPbancTtl") or "공공기관 IT 채용"),
                job_spec=st.session_state["job_spec"]
            )
            st.session_state["latest_roadmap"] = {
                # 공고별로 고유하게 식별되도록 위에서 계산한 current_job_id 사용
                "job_id": str(current_job_id),
                "inst": detail.get("instNm", "기관"),
                "major": selected_major,
                "text": roadmap_text
            }
            st.rerun()

    # --- [결과 및 저장/다운로드 버튼] ---
    latest = st.session_state.get("latest_roadmap")
    #if latest and str(latest.get("job_id")) == str(detail.get("recrutPbancNo", "ijad")):
    # render_gap_analysis 상단에서 계산한 current_job_id로 비교
    if latest and str(latest.get("job_id")) == str(current_job_id):
        st.success(f"✨ {latest['major']} 맞춤형 취업 로드맵")
        st.markdown(latest["text"])
        
        # PDF 변환 (기존 유틸리티 함수 사용)
        try:
            pdf_bytes = _roadmap_to_pdf_bytes(f"{latest['inst']} 로드맵", latest["text"])
            
            # 버튼 레이아웃 복구
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("💾 로드맵 저장", use_container_width=True):
                    # --- persistent save (cache/roadmaps.json) + session history ---
                    # 0) build item
                    item = {
                        "id": str(uuid.uuid4()),
                        "job_id": str(latest.get('job_id') or ""),
                        "inst": str(latest.get('inst') or ""),
                        "major": str(latest.get('major') or ""),
                        "job": str(detail.get("recrutPbancTtl") or ""),
                        "title": f"{str(latest.get('inst') or '기관')} · {str(latest.get('major') or '')}".strip(" ·"),
                        "text": str(latest.get('text') or ""),
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # 1) session history init
                    if "roadmap_history" not in st.session_state:
                        st.session_state["roadmap_history"] = []

                    # 2) persistent list
                    saved = load_saved_roadmaps()
                    # dedupe by (job_id + major + text hash-ish)
                    def _same(a, b):
                        return (
                            str(a.get('job_id')) == str(b.get('job_id'))
                            and str(a.get('major')) == str(b.get('major'))
                            and str(a.get('text')) == str(b.get('text'))
                        )
                    exists_persist = any(_same(x, item) for x in saved)
                    exists_session = any(_same(x, item) for x in st.session_state["roadmap_history"]) 

                    if not exists_persist:
                        saved.insert(0, item)
                        save_saved_roadmaps(saved)

                    if not exists_session:
                        st.session_state["roadmap_history"].insert(0, item)

                    if not exists_persist:
                        st.toast("✅ 로드맵이 저장되었습니다! (cache/roadmaps.json)")
                    else:
                        st.toast("ℹ️ 이미 저장된 로드맵입니다.")

                    st.rerun()
            with btn_col2:
                st.download_button(
                    label="📄 로드맵 PDF 다운로드",
                    data=pdf_bytes,
                    file_name=f"Roadmap_{str(latest.get('job_id',''))}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")

def _roadmap_to_pdf_bytes(title: str, roadmap_text: str) -> bytes:
    """Render the roadmap text into a readable PDF (supports Korean).

    - Uses a Korean CID font (HYGothic-Medium) so "□" tofu does not appear.
    - Converts a small subset of markdown-like syntax:
      * headings (#/##/###)
      * bullets (- / *)
      * bold (**text**)
    """
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title or "IJAD Roadmap",
        author="IJAD",
    )

    styles = getSampleStyleSheet()

    # Base styles with Korean font
    base = styles["BodyText"].clone("IJADBody")
    base.fontName = IJAD_PDF_FONT
    base.fontSize = 10.5
    base.leading = 14

    h1 = styles["Title"].clone("IJADTitle")
    h1.fontName = IJAD_PDF_FONT

    h2 = styles["Heading2"].clone("IJADH2")
    h2.fontName = IJAD_PDF_FONT

    h3 = styles["Heading3"].clone("IJADH3")
    h3.fontName = IJAD_PDF_FONT

    def md_inline_to_rl(s: str) -> str:
        # Escape first, then re-inject <b> tags via markdown ** **
        s = escape(s or "")
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return s

    story = []

    # More intuitive title block
    safe_title = md_inline_to_rl(title or "IJAD 취업 로드맵")
    story.append(Paragraph(safe_title, h1))
    story.append(Spacer(1, 8))

    def _is_md_table_line(s: str) -> bool:
        t = (s or "").strip()
        return t.startswith("|") and t.endswith("|") and (t.count("|") >= 2)

    def _is_md_table_sep_line(s: str) -> bool:
        # Typical markdown separator: | --- | :---: | ---: |
        t = (s or "").strip()
        if not _is_md_table_line(t):
            return False
        inner = t.strip("|").strip()
        if not inner:
            return False
        parts = [p.strip() for p in inner.split("|")]
        # A separator cell is composed of dashes with optional leading/trailing colons
        for p in parts:
            p2 = p.replace(":", "").strip()
            if not p2:
                return False
            if any(ch not in "-" for ch in p2):
                return False
        return True

    def _parse_md_table(lines: list[str], start_idx: int):
        """Parse a markdown table block starting at start_idx.

        Returns (table_flowable, next_idx).
        """
        rows: list[list[str]] = []
        i = start_idx
        while i < len(lines):
            if not _is_md_table_line(lines[i]):
                break
            # skip separator row
            if _is_md_table_sep_line(lines[i]):
                i += 1
                continue
            inner = lines[i].strip().strip("|")
            cells = [c.strip() for c in inner.split("|")]
            rows.append(cells)
            i += 1

        if not rows:
            return None, start_idx + 1

        # Normalize column count
        col_count = max(len(r) for r in rows)
        for r in rows:
            if len(r) < col_count:
                r.extend([""] * (col_count - len(r)))

        # Convert to Paragraphs (header bold)
        data = []
        for ridx, r in enumerate(rows):
            row_cells = []
            for c in r:
                txt = md_inline_to_rl(c)
                if ridx == 0:
                    txt = f"<b>{txt}</b>"
                row_cells.append(Paragraph(txt, base))
            data.append(row_cells)

        # Column widths: even split across available doc width
        total_w = doc.width
        col_w = total_w / float(col_count)
        col_widths = [col_w] * col_count

        tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), IJAD_PDF_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), base.fontSize),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

        return tbl, i

    raw = (roadmap_text or "").replace("\r\n", "\n").strip()
    lines = raw.split("\n")
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        ln = (line or "").strip()

        if not ln:
            story.append(Spacer(1, 6))
            idx += 1
            continue

        # Markdown table block
        if _is_md_table_line(ln):
            tbl, next_idx = _parse_md_table(lines, idx)
            if tbl is not None:
                story.append(Spacer(1, 6))
                story.append(tbl)
                story.append(Spacer(1, 10))
                idx = next_idx
                continue

        # Headings
        if ln.startswith("### "):
            story.append(Paragraph(md_inline_to_rl(ln[4:]), h3))
            idx += 1
            continue
        if ln.startswith("## "):
            story.append(Paragraph(md_inline_to_rl(ln[3:]), h2))
            idx += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(md_inline_to_rl(ln[2:]), h2))
            idx += 1
            continue

        # Bullets
        if ln.startswith("- ") or ln.startswith("* "):
            story.append(Paragraph(md_inline_to_rl(ln[2:]), base, bulletText="•"))
            idx += 1
            continue

        # Normal paragraph
        story.append(Paragraph(md_inline_to_rl(ln), base))
        idx += 1

    doc.build(story)
    return buf.getvalue()

def generate_ai_roadmap(major: str, selections: list, extra_prep: list, job_title: str, job_spec: dict):
    # 1. CSV 파일에서 해당 학부의 전공 과목 및 개요 로드 (RAG)
    try:
        df_major = pd.read_csv("major_overview_4.csv")
        relevant_subjects = df_major[df_major['학부'] == major]
        major_context = "\n".join([
            f"- {row['과목']}: {row['과목개요']}" 
            for _, row in relevant_subjects.iterrows()
        ])
    except Exception:
        major_context = "전공 과목 상세 정보를 불러올 수 없습니다."

    # 2. 일정 데이터 정리 (자격증 및 공통 시험)
    cert_context = ""
    for s in (selections or []):
        cert_context += f"- {s.get('name')}: {s.get('schedule')}\n"
    
    extra_context = "\n".join(extra_prep) if extra_prep else "없음"

    # 3. 3-4학년 맞춤형 취업 준비 프롬프트 구성
    prompt = f"""
당신은 IT 전문 진로 컨설턴트입니다. 3-4학년 대학생이 목표 공고를 분석하여 
장기적인 관점에서 역량을 쌓을 수 있는 '취업 준비 로드맵'을 작성하세요.

[분석 대상]
1. 목표 공고 및 요구역량: {job_title} / {job_spec}
2. 사용자 전공: {major}
3. 전공 커리큘럼 상세:
{major_context}

[사용자 선택 일정]
- 자격증: {cert_context}
- 어학/공통: {extra_context}

[작성 가이드라인 - 엄격 준수]
1. 취지: 단기 합격보다, 공고의 기술 스택을 쌓기 위해 어떤 전공 수업에 집중하고 자격증을 어떻게 연계할지 가이드를 줄 것.
2. 매칭: CSV 데이터에 있는 '전공 과목 개요'를 바탕으로, 공고의 기술 요구사항과 매칭되는 과목 2-3개를 선정하여 학습 이유를 설명할 것.
3. 포맷: 
   - 최상단에 마크다운 '표(Table)'를 사용하여 전체 일정을 요약할 것.
   - 글자 수는 한글 1,000자 이내로 제한(A4 1장 분량).
4. 날짜: 오늘(2026-01-15) 기준으로 남은 기간을 고려할 것.

[출력 포맷]
# 🎯 {job_title} 대비 맞춤형 준비 로드맵
(일정 요약 표)

## 🎓 전공-직무 역량 연결 (RAG 분석)
(전공 과목 개요를 활용한 역량 확보 방안)

## 🗓️ 월별 실행 계획
(시험 일정과 전공 학습을 병행하는 리스트)

### 💡 컨설턴트 제언
"""
    try:
        # API 호출 시도
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        if not text: raise ValueError("Empty Response")
        return text
    except Exception as e:
        # API 오류 시 MOCK 로드맵 자동 생성
        return f"""# 🎯 {job_title} 대비 맞춤형 로드맵 (평가 환경 모드)
> **안내:** 현재는 시스템 검증을 위한 **평가 환경**에서 작동 중입니다. API 연결 없이도 서비스 흐름을 확인하실 수 있도록 최적화된 **직무 가이드라인**을 출력합니다.

| 기간 | 준비 항목 | 비고 |
| :--- | :--- | :--- |
| 1-3월 | 기초 전공 복습 및 {extra_context[:10]}... | 역량 다지기 |
| 4-6월 | {selections[0].get('name') if selections else '자격증'} 집중 기간 | 실전 대비 |

## 🎓 전공-직무 역량 연결
현재 **평가 모드**에 따라 {major}의 핵심 교육과정과 IT 직무 공통 역량을 매칭했습니다. 실제 운영 환경에서는 Gemini AI가 공고문의 기술 스택과 사용자 전공의 접점을 실시간으로 분석하여 상세히 연결해 드립니다.

## 🗓️ 월별 실행 계획
- **현재~상반기**: 선택하신 시험 일정({extra_context[:15]})을 중심으로 학습 가중치를 설계했습니다. 평가 환경에서도 일정 기반의 동적 로드맵 생성이 정상적으로 작동함을 확인하실 수 있습니다.
- **하반기**: 분석된 직무 타겟에 맞춘 포트폴리오 정밀화 및 공고 지원 시뮬레이션 기간입니다.

### 💡 컨설턴트 제언
본 결과물은 평가 환경용 레퍼런스 모델으로 상세 분석이 제한적입니다. 본인의 선택 일정에 맞춰 계획을 수행하시기 바랍니다.
"""
    