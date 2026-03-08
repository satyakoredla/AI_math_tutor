"""
app.py — MathMentor AI: Streamlit-based JEE Math Tutor
Supports text / image / audio input with 5-agent pipeline + HITL + memory.
"""
import os
import sys
import time
import tempfile

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from utils.config import GEMINI_API_KEY, OCR_CONFIDENCE_THRESHOLD, VERIFIER_CONFIDENCE_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MathMentor AI — JEE Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium dark UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.3rem;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

/* ── Agent trace items ── */
.agent-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    background: rgba(167,139,250,0.08);
    border-left: 3px solid #a78bfa;
    margin-bottom: 0.5rem;
    animation: fadeIn 0.4s ease;
}
.agent-icon { font-size: 1.4rem; }
.agent-label { font-weight: 600; color: #a78bfa; font-size: 0.85rem; }
.agent-text  { color: #cbd5e1; font-size: 0.9rem; }

/* ── Answer section ── */
.answer-box {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(96,165,250,0.12));
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.answer-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #34d399;
}

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.conf-high   { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.conf-medium { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
.conf-low    { background: rgba(239,68,68,0.2);   color: #ef4444; border: 1px solid #ef4444; }

/* ── Memory row ── */
.memory-row {
    background: rgba(96,165,250,0.06);
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #94a3b8;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── HITL warning ── */
.hitl-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.4);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Section dividers ── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
}

@keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "input_text": "",
        "ocr_result": None,
        "audio_result": None,
        "hitl_confirmed": False,
        "hitl_text": "",
        "pipeline_result": None,
        "current_memory_id": None,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🧮 MathMentor AI</p>
  <p class="hero-subtitle">Your personal JEE Math Tutor — powered by multi-agent AI</p>
</div>
""", unsafe_allow_html=True)

# API key status badge
if GEMINI_API_KEY:
    st.markdown('<p style="text-align:center;color:#34d399;font-size:0.85rem;">✅ Gemini Connected — Full AI Mode</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="text-align:center;color:#fbbf24;font-size:0.85rem;">⚠️ No API Key — Running with SymPy + Rule-based fallback</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    input_mode = st.radio(
        "**Input Mode**",
        ["✏️ Text", "🖼️ Image", "🎤 Audio"],
        index=0,
        help="Choose how to provide your math problem",
    )

    st.markdown("---")
    show_trace = st.checkbox("🔍 Show Agent Trace", value=True)
    show_docs = st.checkbox("📚 Show Retrieved Formulas", value=True)

    st.markdown("---")
    st.markdown("### 📊 Memory Stats")
    try:
        from memory.memory_store import get_stats, get_recent_problems
        stats = get_stats()
        st.metric("Problems Solved", stats["total_problems"])
        if stats["top_topics"]:
            st.markdown("**Top Topics:**")
            for topic, cnt in stats["top_topics"]:
                st.markdown(f'<div class="memory-row">{topic} — {cnt} solved</div>', unsafe_allow_html=True)
    except Exception:
        st.info("Memory not yet initialised.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **5 Agents:**
    1. 🔍 Parser Agent
    2. 🗺️ Router Agent
    3. 🧮 Solver Agent
    4. ✅ Verifier Agent
    5. 📖 Explainer Agent

    **RAG:** 15 JEE Math docs  
    **Memory:** SQLite  
    **OCR:** EasyOCR  
    **Audio:** Whisper
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<p class="section-title">📥 Input</p>', unsafe_allow_html=True)

    raw_question = ""
    needs_hitl = False
    hitl_reason = ""
    confidence_for_hitl = 1.0

    # ── TEXT MODE ────────────────────────────────────────────────────────────
    if "Text" in input_mode:
        raw_question = st.text_area(
            "Enter your math problem:",
            placeholder="e.g.  Find the derivative of x³ + 2x² − 5x + 3",
            height=140,
            key="text_input_area",
        )

        # Quick example buttons
        st.markdown("**Quick Examples:**")
        ex_cols = st.columns(2)
        examples = [
            "Find the derivative of x^3 + 2x^2 - 5",
            "Solve: 3x + 7 = 22",
            "Integrate x^2 + sin(x)",
            "Find roots of x^2 - 5x + 6 = 0",
        ]
        # Use a separate session state key to avoid widget conflict
        if "example_text" not in st.session_state:
            st.session_state["example_text"] = ""
        for i, ex in enumerate(examples):
            with ex_cols[i % 2]:
                if st.button(ex[:28] + "…", key=f"ex_{i}"):
                    st.session_state["example_text"] = ex
                    st.rerun()
        # Pre-fill text area if example was clicked
        if st.session_state.get("example_text"):
            raw_question = st.session_state["example_text"]
            st.session_state["example_text"] = ""  # clear after use

    # ── IMAGE MODE ───────────────────────────────────────────────────────────
    elif "Image" in input_mode:
        uploaded_img = st.file_uploader(
            "Upload math problem image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="img_upload",
        )
        if uploaded_img:
            from PIL import Image as PILImage
            pil_img = PILImage.open(uploaded_img)
            st.image(pil_img, use_container_width=True, caption="Uploaded Image")

            with st.spinner("🔍 Running OCR..."):
                try:
                    from input_processing.image_ocr import extract_text_from_pil_image
                    ocr_result = extract_text_from_pil_image(pil_img)
                    st.session_state["ocr_result"] = ocr_result
                except Exception as e:
                    ocr_result = {"text": "", "confidence": 0.0, "error": str(e)}
                    st.session_state["ocr_result"] = ocr_result

            ocr = st.session_state.get("ocr_result", {})
            if ocr.get("error"):
                st.warning(f"OCR Error: {ocr['error']}\nPlease install easyocr: `pip install easyocr`")
                raw_question = st.text_area("Or type the problem manually:", height=100, key="img_manual")
            else:
                conf = ocr.get("confidence", 0.0)
                needs_hitl = conf < OCR_CONFIDENCE_THRESHOLD
                confidence_for_hitl = conf
                hitl_reason = f"OCR confidence {conf:.0%} is below threshold. Please verify."
                raw_question = ocr.get("text", "")

    # ── AUDIO MODE ───────────────────────────────────────────────────────────
    elif "Audio" in input_mode:
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload",
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
            with st.spinner("🎙️ Transcribing audio..."):
                try:
                    from input_processing.speech_to_text import transcribe_audio_bytes
                    suffix = "." + uploaded_audio.name.split(".")[-1]
                    asr_result = transcribe_audio_bytes(uploaded_audio.read(), suffix=suffix)
                    st.session_state["audio_result"] = asr_result
                except Exception as e:
                    asr_result = {"text": "", "confidence": 0.0, "error": str(e)}
                    st.session_state["audio_result"] = asr_result

            asr = st.session_state.get("audio_result", {})
            if asr.get("error"):
                st.warning(f"Audio Error: {asr['error']}\nInstall whisper: `pip install openai-whisper`")
                raw_question = st.text_area("Or type the problem manually:", height=100, key="audio_manual")
            else:
                conf = asr.get("confidence", 0.0)
                needs_hitl = conf < 0.75
                confidence_for_hitl = conf
                hitl_reason = f"Transcription confidence {conf:.0%}. Please verify."
                raw_question = asr.get("text", "")

    # ── HITL Panel ───────────────────────────────────────────────────────────
    if raw_question:
        st.markdown("---")
        if needs_hitl:
            st.markdown(f'<div class="hitl-box">⚠️ <b>Human Review Needed:</b> {hitl_reason}</div>', unsafe_allow_html=True)
            st.markdown("")
            hitl_edit = st.text_area(
                "✏️ Review and correct the extracted text:",
                value=raw_question,
                height=100,
                key="hitl_edit",
            )
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("✅ Approve & Solve", key="hitl_approve"):
                    raw_question = hitl_edit
                    st.session_state["hitl_confirmed"] = True
            with col_reject:
                if st.button("✏️ Use Original", key="hitl_use_orig"):
                    st.session_state["hitl_confirmed"] = True
        else:
            if raw_question:
                st.success(f"**Extracted Problem:** {raw_question}")
                st.session_state["hitl_confirmed"] = True

    # ── Check Memory ─────────────────────────────────────────────────────────
    memory_match = None
    if raw_question and len(raw_question) > 5:
        try:
            from memory.memory_store import get_exact_match, find_similar
            memory_match = get_exact_match(raw_question)
        except Exception:
            pass

    if memory_match:
        st.markdown("---")
        st.markdown("**🔁 Found in Memory!**")
        st.info(
            f"This question was solved {memory_match['solve_count']} time(s) before.\n\n"
            f"**Answer:** {memory_match['answer']}"
        )
        use_cached = st.button("⚡ Use Cached Answer", key="use_cached")
        if use_cached:
            st.session_state["pipeline_result"] = {
                "from_memory": True,
                "memory_record": memory_match,
                "question": raw_question,
            }

    # ── Solve Button ─────────────────────────────────────────────────────────
    st.markdown("")
    solve_clicked = st.button(
        "🚀 Solve Problem",
        key="solve_btn",
        use_container_width=True,
        disabled=not bool(raw_question and raw_question.strip()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
if solve_clicked and raw_question and raw_question.strip():
    with col_output:
        st.markdown('<p class="section-title">🤖 Agent Pipeline</p>', unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ── Step 1: Parser Agent ─────────────────────────────────────────────
        status_text.markdown("🔍 **Agent 1: Parsing problem...**")
        progress_bar.progress(15)
        from agents.parser_agent import run_parser_agent
        parsed = run_parser_agent(raw_question)

        if show_trace:
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">🔍</span>
              <div>
                <div class="agent-label">Parser Agent</div>
                <div class="agent-text">Topic: <b>{parsed.get('topic','?')}</b> | Variables: {parsed.get('variables',[])} | Method: {parsed.get('method','?')}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        if parsed.get("needs_clarification"):
            st.warning(f"⚠️ Parser needs clarification: {parsed.get('clarification_reason','')}")

        # ── Step 2: Router Agent ─────────────────────────────────────────────
        status_text.markdown("🗺️ **Agent 2: Routing to solver...**")
        progress_bar.progress(30)
        from agents.router_agent import run_router_agent
        route = run_router_agent(parsed)

        if show_trace:
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">🗺️</span>
              <div>
                <div class="agent-label">Router Agent</div>
                <div class="agent-text">Topic: <b>{route.get('detected_topic','?')}</b> | Mode: {route.get('solver_mode','?')} | Confidence: {route.get('confidence',0):.0%}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Step 3: RAG Retrieval ────────────────────────────────────────────
        status_text.markdown("📚 **Retrieving relevant formulas...**")
        progress_bar.progress(45)
        rag_query = route.get("rag_query", raw_question)
        from rag.retriever import get_relevant_docs, format_docs_for_prompt
        retrieved_docs = get_relevant_docs(rag_query)

        if show_trace:
            doc_sources = ", ".join(d["source"] for d in retrieved_docs)
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">📚</span>
              <div>
                <div class="agent-label">RAG Retriever</div>
                <div class="agent-text">Retrieved {len(retrieved_docs)} doc(s): <b>{doc_sources}</b></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Step 4: Solver Agent ─────────────────────────────────────────────
        status_text.markdown("🧮 **Agent 3: Solving the problem...**")
        progress_bar.progress(60)
        from agents.solver_agent import run_solver_agent
        solver_result = run_solver_agent(parsed, route, retrieved_docs)

        if show_trace:
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">🧮</span>
              <div>
                <div class="agent-label">Solver Agent</div>
                <div class="agent-text">Method: <b>{solver_result.get('method_used','?')}</b> | Confidence: {solver_result.get('confidence',0):.0%}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Step 5: Verifier Agent ───────────────────────────────────────────
        status_text.markdown("✅ **Agent 4: Verifying answer...**")
        progress_bar.progress(75)
        from agents.verifier_agent import run_verifier_agent
        verifier_result = run_verifier_agent(parsed, solver_result)

        if show_trace:
            v_color = "🟢" if verifier_result.get("is_correct") else "🔴"
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">✅</span>
              <div>
                <div class="agent-label">Verifier Agent</div>
                <div class="agent-text">{v_color} Correct: {verifier_result.get('is_correct')} | Confidence: {verifier_result.get('confidence',0):.0%} | HITL: {verifier_result.get('needs_human_review',False)}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Step 6: Explainer Agent ──────────────────────────────────────────
        status_text.markdown("📖 **Agent 5: Generating explanation...**")
        progress_bar.progress(88)
        from agents.explainer_agent import run_explainer_agent
        explainer_result = run_explainer_agent(parsed, solver_result, verifier_result, retrieved_docs)

        if show_trace:
            st.markdown(f"""
            <div class="agent-step">
              <span class="agent-icon">📖</span>
              <div>
                <div class="agent-label">Explainer Agent</div>
                <div class="agent-text">Difficulty: {explainer_result.get('difficulty','?')} | Method: {explainer_result.get('method','?')}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Save to Memory ───────────────────────────────────────────────────
        try:
            from memory.memory_store import save_solution
            mem_id = save_solution(
                raw_question, parsed, retrieved_docs,
                solver_result, explainer_result, verifier_result,
            )
            st.session_state["current_memory_id"] = mem_id
        except Exception:
            pass

        progress_bar.progress(100)
        status_text.markdown("✅ **Pipeline complete!**")
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()

        # Store result
        st.session_state["pipeline_result"] = {
            "from_memory": False,
            "question": raw_question,
            "parsed": parsed,
            "route": route,
            "retrieved_docs": retrieved_docs,
            "solver": solver_result,
            "verifier": verifier_result,
            "explainer": explainer_result,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Output Panel
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("pipeline_result")

if result:
    with col_output:
        if result.get("from_memory"):
            # ── Memory hit display ────────────────────────────────────────────
            mem = result["memory_record"]
            st.markdown('<p class="section-title">🔁 Cached Answer</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="answer-box">
              <p style="color:#94a3b8;font-size:0.9rem;">Previously solved {mem['solve_count']} time(s)</p>
              <p class="answer-value">{mem['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📖 View Full Explanation"):
                st.markdown(mem.get("explanation", "No explanation saved."))

        else:
            verifier = result.get("verifier", {})
            solver = result.get("solver", {})
            explainer = result.get("explainer", {})
            retrieved_docs = result.get("retrieved_docs", [])

            # ── HITL: Verifier not sure ───────────────────────────────────────
            if verifier.get("needs_human_review"):
                st.markdown("""
                <div class="hitl-box">
                  ⚠️ <b>Verifier is unsure about this answer.</b> Please review before using.
                </div>
                """, unsafe_allow_html=True)
                issues = verifier.get("issues", [])
                if issues:
                    for issue in issues:
                        st.warning(f"• {issue}")

            st.markdown("")

            # ── Final Answer ─────────────────────────────────────────────────
            final_ans = verifier.get("verified_answer", solver.get("final_answer", ""))
            conf = verifier.get("confidence", 0.0)
            if conf >= 0.8:
                conf_cls, conf_label = "conf-high", f"✅ {conf:.0%} Confident"
            elif conf >= 0.5:
                conf_cls, conf_label = "conf-medium", f"⚠️ {conf:.0%} Confident"
            else:
                conf_cls, conf_label = "conf-low", f"❌ {conf:.0%} Confident"

            topic_disp = result.get("parsed", {}).get("topic", "math").replace("_", " ").title()
            diff = explainer.get("difficulty", "")

            st.markdown(f"""
            <div class="answer-box">
              <p style="color:#94a3b8;font-size:0.85rem;margin:0">{topic_disp} &nbsp;|&nbsp; {diff}</p>
              <p style="color:#e2e8f0;font-size:1rem;margin:0.3rem 0">Final Answer</p>
              <p class="answer-value">{final_ans}</p>
              <span class="conf-badge {conf_cls}">{conf_label}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Retrieved Formulas ───────────────────────────────────────────
            if show_docs and retrieved_docs:
                with st.expander(f"📚 Retrieved Formulas ({len(retrieved_docs)} docs)", expanded=False):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**[{i}] {doc['source']}**")
                        st.code(doc["content"][:400], language="text")
                        st.markdown("")

            # ── Full Solution ─────────────────────────────────────────────────
            with st.expander("🧮 Full Solution", expanded=True):
                solution_text = solver.get("solution", "No solution generated.")
                st.markdown(solution_text)

            # ── Explanation ───────────────────────────────────────────────────
            with st.expander("📖 Teacher Explanation", expanded=True):
                explanation_text = explainer.get("explanation", "No explanation generated.")
                st.markdown(explanation_text)

                insight = explainer.get("key_insight", "")
                if insight:
                    st.markdown(f'<div class="glass-card">{insight}</div>', unsafe_allow_html=True)

            # ── Feedback ──────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Was this answer helpful?**")
            fb_cols = st.columns(3)
            mem_id = st.session_state.get("current_memory_id")
            with fb_cols[0]:
                if st.button("👍 Correct", key="fb_correct"):
                    if mem_id:
                        from memory.memory_store import update_feedback
                        update_feedback(mem_id, "correct")
                    st.success("Thanks! Feedback saved.")
            with fb_cols[1]:
                if st.button("👎 Incorrect", key="fb_incorrect"):
                    if mem_id:
                        from memory.memory_store import update_feedback
                        update_feedback(mem_id, "incorrect")
                    st.warning("Thanks! We'll note this.")
            with fb_cols[2]:
                correction = st.text_input("✏️ Correct answer:", key="fb_correction", placeholder="Type correction…")
                if correction and st.button("💾 Save", key="fb_save_corr"):
                    if mem_id:
                        from memory.memory_store import update_feedback
                        update_feedback(mem_id, f"correction: {correction}")
                    st.success("Correction saved to memory!")


# ─────────────────────────────────────────────────────────────────────────────
# Recent Memory Section (bottom)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-title">🕐 Recent Problems</p>', unsafe_allow_html=True)

try:
    from memory.memory_store import get_recent_problems
    recent = get_recent_problems(limit=6)
    if recent:
        cols = st.columns(3)
        for i, prob in enumerate(recent):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="memory-row">
                  <b>{prob['topic'].replace('_',' ').title()}</b><br>
                  {prob['question'][:55]}…<br>
                  <span style="color:#60a5fa">Ans: {prob['answer'][:30]}</span><br>
                  <small style="color:#64748b">{prob['timestamp']} · ×{prob['solve_count']}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#475569;text-align:center">No problems solved yet. Ask your first question above!</p>', unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding:1rem;color:#475569;font-size:0.8rem;">
  MathMentor AI · Built for JEE · 5-Agent Pipeline · RAG + Memory + HITL
</div>
""", unsafe_allow_html=True)
