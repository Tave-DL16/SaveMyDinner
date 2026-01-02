"""
modules.UI.streamlit_app의 Docstring
김수민
스트림릿
"""
import streamlit as st
from typing import Any, Optional, List, Dict

from modules.ui.services import detect_ingredients, get_dish_candidates, get_recipe_links


# -----------------------------
# Session State Management
# -----------------------------
def init_state() -> None:
    """
    Streamlit Session State 초기화 함수
    - 앱 실행 시 필요한 기본 변수들을 설정합니다.
    """
    defaults = {
        "step": 1,
        "theme": "dark",  # "dark" | "light"

        "image_file": None,
        "ingredients": [],
        "dish_candidates": [],
        "selected_dish": None,
        "final_result": None,

        # Step3 widget states (중요: 사이드바 즉시 반영용)
        "dish_radio": None,  # 추천 목록 선택값
        "dish_custom": "",  # 직접 입력값

        # 에러 상태 관리
        "detect_failed": False,
        "detect_error_msg": None,
        "candidate_failed": False,
        "candidate_error_msg": None,
        "links_failed": False,
        "links_error_msg": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_step(n: int) -> None:
    """
    페이지 단계(Step)를 이동시키는 함수
    Input: n (이동할 단계 번호)
    """
    st.session_state.step = n


def reset_all() -> None:
    """
    전체 상태를 초기화하는 함수 (테마 설정은 유지)
    - '다시 시작하기' 기능에 사용됩니다.
    """
    # 테마 설정을 저장
    current_theme = st.session_state.get("theme", "dark")
    st.session_state.clear()
    # 테마 복원
    st.session_state.theme = current_theme


# -----------------------------
# Theme & UI Rendering
# -----------------------------
def _theme_tokens(theme: str) -> Dict[str, str]:
    """
    선택된 테마(light/dark)에 따른 CSS 색상 토큰을 반환하는 함수
    Input: theme (str) - "light" or "dark"
    Output: dict - CSS 색상 코드 딕셔너리
    """
    if theme == "light":
        return {
            "bg": "#f8fafc",  # 부드러운 회색빛 흰색
            "text": "#0f172a",  # 진한 남색 (높은 대비)
            "muted": "#64748b",  # 중간 회색 (caption용)
            "card": "#ffffff",  # 순백 (카드 배경)
            "border": "#e2e8f0",  # 연한 회색 테두리
            "accent": "#3b82f6",  # 생생한 파란색 (primary)
            "accent2": "#60a5fa",  # 조금 더 밝은 파란색 (secondary)
            "danger": "#ef4444",  # 선명한 빨간색
            "shadow": "rgba(15, 23, 42, 0.08)",
            "hover_shadow": "rgba(59, 130, 246, 0.15)",
        }
    # dark mode defaults
    return {
        "bg": "#0b1220",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "card": "#101b33",
        "border": "#243453",
        "accent": "#3b82f6",  # 파란색 (Progress Bar 통일)
        "accent2": "#60a5fa",  # 밝은 파란색 (Gradient)
        "danger": "#FF6B6B",
        "shadow": "rgba(0,0,0,0.35)",
        "hover_shadow": "rgba(59, 130, 246, 0.25)",
    }


def render_header() -> None:
    """
    앱의 헤더와 공통 CSS 스타일(Custom CSS)을 렌더링하는 함수
    - 테마 토큰을 기반으로 동적 스타일링을 적용합니다.
    """
    t = _theme_tokens(st.session_state.theme)

    # ✅ 라이트 모드 문제 해결용 CSS
    light_fix_css = ""
    if st.session_state.theme == "light":
        light_fix_css = f"""
        /* ===== [LIGHT FIX] Step1 FileUploader dropzone background & text ===== */
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
            background-color: {t["card"]} !important;
        }}
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div {{
            background-color: {t["card"]} !important;
        }}
        section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {{
            color: {t["text"]} !important;
        }}
        /* ===== [LIGHT FIX] Uploaded filename text -> muted gray ===== */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] * {{
            color: {t["muted"]} !important;
        }}
        /* ===== [LIGHT FIX] Step2 "추가하기" submit box/button ===== */
        div[data-testid="stForm"] button,
        div[data-testid="stForm"] button[type="submit"],
        div[data-testid="stForm"] .stButton button {{
            background-color: {t["accent"]} !important;
            color: #ffffff !important;
            border: 1px solid {t["accent"]} !important;
        }}
        div[data-testid="stForm"] button:hover,
        div[data-testid="stForm"] button[type="submit"]:hover,
        div[data-testid="stForm"] .stButton button:hover {{
            background-color: {t["accent2"]} !important;
            border-color: {t["accent2"]} !important;
        }}
        """

    st.markdown(
        f"""
        <style>
        /* ===== Page background & base text ===== */
        html, body, [data-testid="stAppViewContainer"] {{
            background: {t["bg"]} !important;
            color: {t["text"]} !important;
        }}
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* markdown 기본 텍스트 */
        .stMarkdown, .stCaption, .stText, p, span, label {{
            color: {t["text"]} !important;
        }}
        .stCaption {{
            color: {t["muted"]} !important;
        }}

        /* ===== Buttons ===== */
        .stButton button {{
            transition: all 0.25s ease;
            border: 1px solid {t["border"]} !important;
            background-color: {t["card"]} !important;
            color: {t["text"]} !important;
        }}
        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px {t["hover_shadow"]};
            border-color: {t["accent"]} !important;
        }}
        .stButton button:active {{
            transform: translateY(0px);
        }}

        /* Primary button style */
        .stButton button[kind="primary"] {{
            background: linear-gradient(135deg, {t["accent"]} 0%, {t["accent2"]} 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600;
        }}

        /* ===== Containers (cards) ===== */
        div[data-testid="stContainer"] {{
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            background-color: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
        }}
        div[data-testid="stContainer"]:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px {t["shadow"]};
        }}

        /* ===== Progress bar (Blue Gradient) ===== */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {t["accent"]} 0%, {t["accent2"]} 100%) !important;
        }}

        /* ===== Inputs / Selects / Radios ===== */
        /* text input */
        div[data-testid="stTextInput"] input {{
            border: 1.5px solid {t["border"]} !important;
            background: {t["card"]} !important;
            color: {t["text"]} !important;
            transition: all 0.2s ease;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border: 2px solid {t["accent"]} !important;
            box-shadow: 0 0 0 3px {t["hover_shadow"]};
        }}
        div[data-testid="stTextInput"] input[aria-invalid="true"] {{
            border: 2px solid {t["danger"]} !important;
            box-shadow: none !important;
        }}

        /* multiselect/selectbox */
        div[data-baseweb="select"] > div {{
            background: {t["card"]} !important;
            color: {t["text"]} !important;
            border-color: {t["border"]} !important;
        }}
        div[data-baseweb="select"] div[role="button"] {{
            color: {t["text"]} !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {t["text"]} !important;
        }}

        /* Multiselect 태그(칩) 색상: Accent Color */
        span[data-baseweb="tag"] {{
            background-color: {t["accent"]} !important;
            color: #ffffff !important;
        }}
        /* Multiselect 포커스 시 테두리 색상 */
        div[data-baseweb="select"] > div:focus-within {{
            border-color: {t["accent"]} !important;
            box-shadow: 0 0 0 2px {t["hover_shadow"]} !important;
        }}

        /* Radio buttons */
        div[data-testid="stRadio"] label {{
            color: {t["text"]} !important;
            background-color: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }}
        div[data-testid="stRadio"] label:hover {{
            border-color: {t["accent"]} !important;
            box-shadow: 0 2px 8px {t["hover_shadow"]};
        }}

        /* ===== Alerts ===== */
        .stAlert {{
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid {t["border"]} !important;
        }}
        .stAlert [data-testid="stMarkdownContainer"] {{
            color: {t["text"]} !important;
        }}

        /* Info alert */
        div[data-baseweb="notification"][kind="info"] {{
            background-color: {t["card"]} !important;
            border-left: 4px solid {t["accent"]} !important;
        }}

        /* Success alert */
        div[data-baseweb="notification"][kind="success"] {{
            background-color: {t["card"]} !important;
            border-left: 4px solid #10b981 !important;
        }}

        /* Warning alert */
        div[data-baseweb="notification"][kind="warning"] {{
            background-color: {t["card"]} !important;
            border-left: 4px solid #f59e0b !important;
        }}

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {{
            background-color: {t["card"]} !important;
            border-right: 1px solid {t["border"]} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {t["text"]} !important;
        }}

        /* ===== Divider ===== */
        hr {{
            border-color: {t["border"]} !important;
        }}

        /* ===== Metric ===== */
        div[data-testid="stMetric"] {{
            background-color: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
            border-radius: 0.5rem;
            padding: 1rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {t["muted"]} !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {t["accent"]} !important;
        }}

        /* ===== File Uploader ===== */
        section[data-testid="stFileUploader"] {{
            border: 2px dashed {t["border"]} !important;
            background-color: {t["card"]} !important;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }}
        section[data-testid="stFileUploader"]:hover {{
            border-color: {t["accent"]} !important;
            box-shadow: 0 4px 12px {t["hover_shadow"]};
        }}
        section[data-testid="stFileUploader"] > div {{ background-color: {t["card"]} !important; }}
        section[data-testid="stFileUploader"] > div > div {{ background-color: {t["card"]} !important; }}
        section[data-testid="stFileUploader"] * {{ color: {t["text"]} !important; }}
        section[data-testid="stFileUploader"] label {{ color: {t["text"]} !important; }}

        /* Browse files 버튼 */
        section[data-testid="stFileUploader"] button {{
            background-color: {t["accent"]} !important;
            color: white !important;
            border: 1px solid {t["accent"]} !important;
        }}
        section[data-testid="stFileUploader"] button:hover {{
            background-color: {t["accent2"]} !important;
            border-color: {t["accent2"]} !important;
        }}
        section[data-testid="stFileUploader"] small {{ color: {t["muted"]} !important; }}
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {{
            background-color: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
        }}

        /* ===== Form Submit Button ===== */
        .stForm button[kind="primary"],
        .stForm button[type="submit"] {{
            background-color: {t["accent"]} !important;
            color: white !important;
            border: 1px solid {t["accent"]} !important;
        }}
        .stForm button[kind="primary"]:hover,
        .stForm button[type="submit"]:hover {{
            background-color: {t["accent2"]} !important;
            border-color: {t["accent2"]} !important;
        }}

        /* ===== Helper Classes ===== */
        .smd-tip {{
            width: 100%;
            text-align: right;
            margin-top: 0.35rem;
            color: {t["muted"]};
            font-size: 0.92rem;
            white-space: nowrap;
        }}

        /* ===== Link buttons ===== */
        .stLinkButton a {{
            background: linear-gradient(135deg, {t["accent"]} 0%, {t["accent2"]} 100%) !important;
            color: white !important;
            text-decoration: none !important;
            border: none !important;
            transition: all 0.25s ease !important;
        }}
        .stLinkButton a:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px {t["hover_shadow"]} !important;
        }}

        /* ===== Expander & Toast ===== */
        div[data-testid="stExpander"] {{
            background-color: {t["card"]} !important;
            border: 1px solid {t["border"]} !important;
        }}
        div[data-testid="stExpander"] summary {{ color: {t["text"]} !important; }}

        .stToast {{
            background-color: {t["card"]} !important;
            color: {t["text"]} !important;
            border: 1px solid {t["border"]} !important;
        }}

        {light_fix_css}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("SaveMyDinner 저메추 🍽️")
    st.caption("냉장고/재료 사진을 업로드하면 재료 인식 → 요리 추천 → 레시피 링크까지!")
    st.divider()


def render_sidebar() -> None:
    """
    사이드바(테마 선택, 진행 상황, 고급 설정 등)를 렌더링하는 함수
    """
    with st.sidebar:
        st.markdown("### 🎛️ 테마")

        prev_theme = st.session_state.get("theme", "dark")

        current_theme = st.radio(
            "모드 선택",
            options=["dark", "light"],
            index=0 if st.session_state.theme == "dark" else 1,
            key="theme_selector",
            label_visibility="collapsed",
        )

        if current_theme != prev_theme:
            st.session_state.theme = current_theme
            st.rerun()

        st.divider()
        st.markdown("### 📋 사용 가이드")

        current_step = st.session_state.step
        steps = [
            ("📸", "사진 업로드"),
            ("🥕", "재료 확인"),
            ("🍳", "요리 선택"),
            ("📱", "추천 결과"),
        ]

        for i, (icon, label) in enumerate(steps, 1):
            if i == current_step:
                st.markdown(f"**{icon} {i}. {label}** ← 현재")
            elif i < current_step:
                st.markdown(f"~~{icon} {i}. {label}~~ ✅")
            else:
                st.markdown(f"{icon} {i}. {label}")

        st.divider()

        if st.session_state.ingredients:
            st.metric("현재 재료 수", len(st.session_state.ingredients))

        if st.session_state.selected_dish:
            st.info(f"선택한 요리: **{st.session_state.selected_dish}**")

        st.divider()

        if st.button("🔄 전체 초기화", use_container_width=True, type="secondary"):
            reset_all()
            st.rerun()

        with st.expander("⚙️ 고급 설정"):
            debug = st.toggle("디버그 모드", value=False)
            if debug:
                st.json(dict(st.session_state), expanded=False)


def render_progress() -> None:
    """
    상단 진행 바(Progress Bar)를 렌더링하는 함수
    """
    step = st.session_state.step
    st.progress((step - 1) / 3)
    labels = {1: "사진 업로드", 2: "재료 확인", 3: "요리 선택", 4: "추천 결과"}
    st.subheader(f"STEP {step}. {labels[step]}")


# -----------------------------
# Error Handling Wrappers
# -----------------------------
def reset_error_state(error_type: str) -> None:
    """
    API 호출 전, 특정 단계의 에러 상태를 초기화하는 함수
    Input: error_type (str) - 'detect', 'candidate', 'links'
    """
    error_mapping = {
        "detect": ("detect_failed", "detect_error_msg"),
        "candidate": ("candidate_failed", "candidate_error_msg"),
        "links": ("links_failed", "links_error_msg"),
    }

    if error_type in error_mapping:
        failed_key, msg_key = error_mapping[error_type]
        st.session_state[failed_key] = False
        st.session_state[msg_key] = None


def handle_api_call(api_func, *args, error_type: str, **kwargs) -> Any:
    """
    API 함수를 안전하게 호출하고 예외 발생 시 에러 상태를 업데이트하는 래퍼 함수
    Input: api_func (function), error_type (str), *args, **kwargs
    Output: Any (성공 시 결과값, 실패 시 빈 리스트 또는 딕셔너리 반환)
    """
    reset_error_state(error_type)

    try:
        result = api_func(*args, **kwargs)
        return result if result else []
    except Exception as e:
        error_mapping = {
            "detect": ("detect_failed", "detect_error_msg"),
            "candidate": ("candidate_failed", "candidate_error_msg"),
            "links": ("links_failed", "links_error_msg"),
        }

        if error_type in error_mapping:
            failed_key, msg_key = error_mapping[error_type]
            st.session_state[failed_key] = True
            st.session_state[msg_key] = str(e)

        return [] if error_type != "links" else {}


def render_error_message(error_type: str, custom_message: Optional[str] = None) -> None:
    """
    에러 발생 시 사용자에게 경고 메시지를 표시하는 함수
    Input: error_type (str), custom_message (str | None)
    """
    error_info = {
        "detect": (
            "detect_failed",
            "detect_error_msg",
            "재료 인식에 실패했어요 😥 아래 **'재료 직접 추가'**로 재료를 입력해 주세요.",
        ),
        "candidate": (
            "candidate_failed",
            "candidate_error_msg",
            "요리 후보 검색에 실패했어요. 재료를 다시 확인하거나 직접 요리를 입력해 주세요.",
        ),
        "links": (
            "links_failed",
            "links_error_msg",
            "레시피 링크를 가져오지 못했어요. 잠시 후 다시 시도하거나, 다른 요리를 선택해 주세요.",
        ),
    }

    if error_type not in error_info:
        return

    failed_key, msg_key, default_message = error_info[error_type]

    if st.session_state.get(failed_key):
        st.warning(custom_message or default_message)
        if st.session_state.get(msg_key):
            st.caption(f"에러 정보(디버그용): {st.session_state[msg_key]}")


# -----------------------------
# Step 1: Upload
# -----------------------------
def step1_upload() -> None:
    """
    [STEP 1] 이미지 업로드 화면을 렌더링하고 재료 인식 API를 호출하는 함수
    Output: None (UI Render)
    """
    col1, col2 = st.columns([3, 2])
    with col1:
        st.info("📷 냉장고나 재료 사진을 업로드해주세요!")
    with col2:
        st.markdown('<div class="smd-tip">💡 Tip: 밝은 곳에서 찍은 선명한 사진이 좋아요</div>', unsafe_allow_html=True)

    file = st.file_uploader(
        "사진 업로드",
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, JPEG 형식만 지원합니다",
    )

    if file:
        st.session_state.image_file = file

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(file, caption="✨ 업로드된 이미지", use_container_width=True)

        st.success("✅ 이미지 업로드 완료! 다음 버튼을 눌러주세요.")

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ 다음 단계: 재료 인식하기", use_container_width=True, disabled=not file):
            with st.spinner("🔍 AI가 재료를 인식하는 중..."):
                ingredients = handle_api_call(detect_ingredients, file, error_type="detect")

            st.session_state.ingredients = ingredients

            if not ingredients:
                st.session_state.detect_failed = True

            go_step(2)
            st.rerun()


def render_ingredient_highlight_css() -> None:
    """
    재료 인식 실패 시 수동 입력창 강조 애니메이션 CSS 주입
    """
    t = _theme_tokens(st.session_state.theme)
    highlight_color = "#f59e0b" if st.session_state.theme == "light" else "#FFC107"

    st.markdown(
        f"""
        <style>
        @keyframes pulseBorder {{
          0%   {{ box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.8); }}
          70%  {{ box-shadow: 0 0 0 12px rgba(245, 158, 11, 0.0); }}
          100% {{ box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.0); }}
        }}
        div[data-testid="stTextInput"] input[aria-label="재료 직접 추가"] {{
          border: 2px solid {highlight_color} !important;
          animation: pulseBorder 1.2s ease-out 2;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Step 2: Ingredients
# -----------------------------
def step2_ingredients() -> None:
    """
    [STEP 2] 인식된 재료 목록을 확인, 수정, 추가하는 화면 함수
    Output: None (UI Render)
    """
    if st.session_state.detect_failed and not st.session_state.ingredients:
        render_error_message("detect")
        render_ingredient_highlight_css()
    else:
        st.success("✅ 재료 인식 완료! 필요하면 수정하거나 추가해주세요.")

    if st.session_state.ingredients:
        st.info(f"🥕 현재 **{len(st.session_state.ingredients)}개**의 재료가 선택되었어요")

    detected = st.session_state.ingredients
    base_options = sorted(
        set(detected + ["감자", "당근", "두부", "김치", "버터", "닭가슴살", "양파", "마늘", "대파", "고추"])
    )

    st.markdown("#### 📝 재료 선택 및 수정")
    edited = st.multiselect(
        "재료 목록",
        options=base_options,
        default=detected,
        help="클릭해서 재료를 추가하거나 X 버튼으로 제거할 수 있어요",
    )
    st.session_state.ingredients = edited

    st.divider()

    st.markdown("#### ➕ 재료 직접 추가")
    with st.form("add_ing_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ing = st.text_input(
                "재료명",
                placeholder="예: 김치, 버터, 닭가슴살",
                key="manual_ing_input",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("추가하기", use_container_width=True)

    if submitted and new_ing:
        new_ing = new_ing.strip()
        if new_ing and new_ing not in st.session_state.ingredients:
            st.session_state.ingredients.append(new_ing)
            st.toast(f"✅ '{new_ing}' 추가 완료!", icon="✅")
            st.rerun()
        elif new_ing in st.session_state.ingredients:
            st.toast(f"⚠️ '{new_ing}'는 이미 있어요!", icon="⚠️")

    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            go_step(1)
            st.rerun()

    with col3:
        is_disabled = len(st.session_state.ingredients) == 0
        if st.button("➡️ 다음: 요리 추천받기", use_container_width=True, disabled=is_disabled):
            with st.spinner("🔍 요리 후보를 찾는 중..."):
                candidates = handle_api_call(
                    get_dish_candidates,
                    st.session_state.ingredients,
                    error_type="candidate",
                )

            st.session_state.dish_candidates = candidates
            go_step(3)
            st.rerun()


# -----------------------------
# Step 3: Choose Dish
# -----------------------------
def step3_choose_dish() -> None:
    """
    [STEP 3] 추천된 요리 목록 중 하나를 선택하거나 직접 입력하는 화면 함수
    Output: None (UI Render)
    """
    if st.session_state.candidate_failed:
        render_error_message("candidate")
    else:
        st.info("🍳 추천 요리 중에서 선택하거나, 직접 입력해주세요!")

    candidates = (st.session_state.dish_candidates or [])[:3]

    if candidates:
        st.markdown("#### 🎯 AI 추천 요리")

        st.radio(
            "추천 목록에서 선택",
            candidates,
            index=None,
            key="dish_radio",
            label_visibility="collapsed",
        )
    else:
        st.warning("추천 요리가 없어요 😥")

    st.divider()

    st.markdown("#### 💭 원하는 요리를 직접 입력")
    st.text_input(
        "요리명",
        placeholder="예: 김치찌개, 된장찌개, 계란볶음밥",
        key="dish_custom",
        label_visibility="collapsed",
    )

    final_choice = st.session_state.selected_dish or ""
    if final_choice:
        st.success(f"✅ 현재 선택: **{final_choice}**")

    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            go_step(2)
            st.rerun()

    with col3:
        is_disabled = not bool(final_choice)
        if st.button("✨ 레시피 찾기", use_container_width=True, disabled=is_disabled):
            with st.spinner("🔍 맞춤 레시피를 찾는 중..."):
                result = handle_api_call(get_recipe_links, final_choice, error_type="links")

            st.session_state.final_result = result

            if not result:
                st.session_state.links_failed = True

            go_step(4)
            st.rerun()


# -----------------------------
# Step 4: Results
# -----------------------------
def render_link_card(title: str, subtitle: str, url: str, index: int) -> None:
    """
    개별 유튜브 영상 링크 카드를 생성하는 함수
    Input: title, subtitle, url, index
    """
    with st.container(border=True):
        rank_emoji = ["🥇", "🥈", "🥉"][index] if index < 3 else "📌"
        st.markdown(f"### {rank_emoji} {title}")

        if subtitle:
            st.caption(f"👤 {subtitle}")

        st.write("")
        st.link_button("🎬 영상 보러가기", url, use_container_width=True)
        st.caption("💡 새 탭에서 열립니다")


def render_retry_buttons(dish: str) -> None:
    """
    에러 발생 시 재시도 버튼 그룹을 렌더링하는 함수
    """
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔁 링크 다시 가져오기", use_container_width=True, disabled=not dish):
            with st.spinner("추천 레시피 찾는 중..."):
                result = handle_api_call(get_recipe_links, dish, error_type="links")
            st.session_state.final_result = result
            st.rerun()

    with col2:
        if st.button("⬅️ 요리 다시 선택", use_container_width=True):
            go_step(3)
            st.rerun()


def render_navigation_buttons() -> None:
    """
    결과 화면 하단의 네비게이션 버튼(이전, 처음으로)을 렌더링하는 함수
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ 이전 (요리 다시 선택)", use_container_width=True):
            go_step(3)
            st.rerun()

    with col3:
        if st.button("🏠 처음으로", use_container_width=True):
            reset_all()
            init_state()
            st.rerun()


def step4_results() -> None:
    """
    [STEP 4] 최종 결과(유튜브 레시피 링크) 화면을 렌더링하는 함수
    Output: None (UI Render)
    """
    dish = st.session_state.selected_dish
    result = st.session_state.final_result or {}

    st.markdown(f"### 🎉 선택한 요리: **{dish}**")
    st.caption("AI가 선별한 최고의 레시피 영상을 확인해보세요!")
    st.divider()

    if st.session_state.links_failed or not result:
        render_error_message("links")
        render_retry_buttons(dish)
        st.divider()

    st.subheader("🎬 추천 레시피 영상")

    youtube_list = (result.get("youtube") or [])[:3]

    if not youtube_list:
        st.info("😥 표시할 레시피 영상이 없어요")
        with st.expander("💡 이럴 때는?"):
            st.write(
                """
                - 다른 요리명으로 다시 시도해보세요
                - 좀 더 일반적인 요리명을 사용해보세요
                - 예: '계란볶음밥' 대신 '볶음밥'
                """
            )
        st.write("")
        render_navigation_buttons()
        return

    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            if i < len(youtube_list):
                item = youtube_list[i]
                title = item.get("title", "제목 없음")
                channel = item.get("channel", "")
                url = item.get("url", "")
                if url:
                    render_link_card(title, channel, url, i)
                else:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption("⚠️ 링크가 없어요")

    st.divider()
    st.write("")
    render_navigation_buttons()


# -----------------------------
# Step3 selection sync
# -----------------------------
def sync_selected_dish_from_widgets() -> None:
    """
    위젯(라디오 버튼, 텍스트 입력)의 값을 세션 상태와 동기화하는 함수
    """
    custom = (st.session_state.get("dish_custom") or "").strip()
    radio = st.session_state.get("dish_radio")

    final_choice = custom if custom else (radio or "")
    st.session_state.selected_dish = final_choice if final_choice else None


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """
    메인 애플리케이션 실행 함수
    - 페이지 설정, 초기화, 사이드바, 단계별 렌더링을 순차적으로 호출합니다.
    """
    st.set_page_config(page_title="SaveMyDinner", page_icon="🍽️", layout="wide")

    init_state()

    sync_selected_dish_from_widgets()

    render_header()

    render_sidebar()

    render_progress()

    step_functions = {
        1: step1_upload,
        2: step2_ingredients,
        3: step3_choose_dish,
        4: step4_results,
    }

    current_step = st.session_state.step
    if current_step in step_functions:
        step_functions[current_step]()


if __name__ == "__main__":
    main()
