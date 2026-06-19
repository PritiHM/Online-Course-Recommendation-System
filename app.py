"""
app.py — Course Recommendation System (Streamlit, Coursera-style)
Keyword search → hybrid recommendations with topic-aware similarity.
"""

import pickle
import re
import numpy as np
import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="CourseMatch", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── BACKGROUND ── */
.stApp {
    background: #0d0f1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,51,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,212,180,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 60% 20%, rgba(255,77,141,0.08) 0%, transparent 50%);
}

.block-container {
    max-width: 1280px;
    padding-top: 0rem !important;
}

/* ── NAVBAR ── */
.nav-bar {
    background: rgba(15,17,32,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    height: 72px;
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0 2rem;
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

/* ── HERO ── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #1a1040 0%, #0d1f3c 50%, #0a2520 100%);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 4.5rem 2rem 3.5rem;
    border-radius: 28px;
    margin: 1.5rem 0 2rem 0;
    text-align: center;
    box-shadow: 0 40px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
}

.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.4) 0%, transparent 70%);
    pointer-events: none;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 300px; height: 150px;
    background: radial-gradient(ellipse, rgba(0,212,180,0.2) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
}

.hero-title span {
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── SEARCH BAR ── */
.stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    color: #f1f5f9 !important;
    border: 1.5px solid rgba(167,139,250,0.35) !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 0 0 0 transparent, inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: all 0.25s ease !important;
}

.stTextInput input:focus {
    border-color: #a78bfa !important;
    background: rgba(167,139,250,0.08) !important;
    box-shadow: 0 0 0 4px rgba(167,139,250,0.12) !important;
}

.stTextInput input::placeholder {
    color: rgba(255,255,255,0.3) !important;
}

/* ── BUTTONS ── */
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    height: 50px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.25s ease !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.55) !important;
}

/* ── COURSE CARD ── */
.course-card {
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 4px solid;
    border-left-color: #7c3aed;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.course-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}

.course-card:hover {
    transform: translateY(-4px);
    background: rgba(255,255,255,0.07);
    border-color: rgba(167,139,250,0.3);
    border-left-color: #a78bfa;
    box-shadow: 0 16px 48px rgba(0,0,0,0.4), 0 0 0 1px rgba(167,139,250,0.15);
}

.course-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
}

.course-inst {
    color: rgba(255,255,255,0.45);
    font-size: 0.875rem;
    margin-top: 3px;
}

.course-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 10px 0 14px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
}

.course-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    font-size: 1.3rem;
    float: right;
    margin-left: 12px;
}

.course-body {
    overflow: hidden;
}

/* ── BADGES ── */
.pill {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.pill-green {
    background: rgba(52,211,153,0.15);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
}

.pill-yellow {
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25);
}

.pill-red {
    background: rgba(248,113,113,0.15);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.25);
}

.pill-purple {
    background: rgba(167,139,250,0.15);
    color: #a78bfa;
    border: 1px solid rgba(167,139,250,0.25);
}

/* ── STARS ── */
.stars {
    color: #fbbf24;
    letter-spacing: 1px;
}

/* ── MATCH BAR ── */
.match-bar-bg {
    background: rgba(255,255,255,0.08);
    height: 6px;
    border-radius: 20px;
    margin-top: 4px;
}

.match-bar-fg {
    background: linear-gradient(90deg, #34d399, #38bdf8, #a78bfa);
    height: 6px;
    border-radius: 20px;
}

.match-label {
    color: rgba(255,255,255,0.35);
    font-size: 0.75rem;
    margin-top: 6px;
}

/* ── FILTER ROW ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 1.5rem 0 0.25rem;
    letter-spacing: -0.02em;
}

.section-sub {
    color: rgba(255,255,255,0.4);
    font-size: 0.875rem;
    margin-bottom: 1.25rem;
}

/* ── EMPTY / NO RESULTS ── */
.empty-state, .no-results {
    text-align: center;
    padding: 5rem 1rem;
}

.empty-icon, .no-results-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 1rem;
}

.empty-title, .no-results-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}

.empty-sub, .no-results-sub {
    color: rgba(255,255,255,0.4);
    font-size: 0.9rem;
}

/* ── STREAMLIT OVERRIDES ── */
.stMarkdown, .stText, p, label, div {
    color: #e2e8f0;
}

.stInfo {
    background: rgba(56,189,248,0.1) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    color: #7dd3fc !important;
}

</style>
""", unsafe_allow_html=True)


# ── Topic taxonomy ─────────────────────────────────────────────────────────────
# Each course mapped to topic tags. Similarity = Jaccard over shared tags.
COURSE_TOPICS = {
    "AI for Business Leaders":                  {"ai", "business", "management", "tech"},
    "Advanced Machine Learning":                {"ai", "ml", "data", "tech", "programming"},
    "Blockchain and Decentralized Applications":{"blockchain", "crypto", "tech", "programming", "finance"},
    "Cloud Computing Essentials":               {"cloud", "tech", "infrastructure", "devops"},
    "Cybersecurity for Professionals":          {"security", "tech", "networking", "hacking"},
    "Data Visualization with Tableau":          {"data", "analytics", "visualization", "business"},
    "DevOps and Continuous Deployment":         {"devops", "tech", "infrastructure", "programming", "cloud"},
    "Ethical Hacking Masterclass":              {"security", "hacking", "tech", "networking"},
    "Fitness and Nutrition Coaching":           {"health", "fitness", "lifestyle", "wellness"},
    "Fundamentals of Digital Marketing":        {"marketing", "business", "digital", "seo"},
    "Game Development with Unity":              {"programming", "gamedev", "tech", "creative"},
    "Graphic Design with Canva":                {"design", "creative", "visual", "art"},
    "Mobile App Development with Swift":        {"programming", "mobile", "tech", "ios"},
    "Networking and System Administration":     {"networking", "tech", "infrastructure", "security"},
    "Personal Finance and Wealth Building":     {"finance", "money", "investing", "lifestyle"},
    "Photography and Video Editing":            {"photography", "creative", "art", "visual", "media"},
    "Project Management Fundamentals":          {"management", "business", "agile", "planning"},
    "Public Speaking Mastery":                  {"communication", "business", "soft-skills", "presentation"},
    "Python for Beginners":                     {"programming", "python", "tech", "data"},
    "Stock Market and Trading Strategies":      {"finance", "investing", "trading", "money"},
}

def build_topic_sim(all_courses: list) -> pd.DataFrame:
    """Jaccard similarity matrix based on shared topic tags."""
    n = len(all_courses)
    mat = np.zeros((n, n))
    for i, a in enumerate(all_courses):
        for j, b in enumerate(all_courses):
            if i == j:
                mat[i, j] = 1.0
                continue
            ta, tb = COURSE_TOPICS.get(a, set()), COURSE_TOPICS.get(b, set())
            inter = len(ta & tb)
            union = len(ta | tb)
            mat[i, j] = inter / union if union else 0.0
    return pd.DataFrame(mat, index=all_courses, columns=all_courses)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    with open(config.MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)

    cos_sim_df      = bundle["cos_sim_df"]
    course_profiles = bundle["course_profiles"]
    knn_model       = bundle["knn_model"]
    df              = bundle["preprocessed_df"]
    all_courses     = cos_sim_df.index.tolist()

    interaction = (
        df.pivot_table(index="user_id", columns="course_name", values="rating", aggfunc="mean")
        .fillna(0)
        .reindex(columns=all_courses, fill_value=0)
    )
    course_mat = interaction.T

    stats = df.groupby("course_name").agg(
        rating=("rating", "mean"),
        enrollment_numbers=("enrollment_numbers", "mean"),
        course_price=("course_price", "mean"),
        course_duration_hours=("course_duration_hours", "mean"),
    ).round(2)

    topic_sim = build_topic_sim(all_courses)

    return {
        "knn": knn_model,
        "profiles": course_profiles,
        "cos_sim": cos_sim_df,
        "topic_sim": topic_sim,
        "course_mat": course_mat,
        "all_courses": all_courses,
        "stats": stats,
    }


# ── Keyword matching ───────────────────────────────────────────────────────────
KEYWORD_MAP = {
    "python":             ["Python for Beginners"],
    "ml":                 ["Advanced Machine Learning"],
    "machine learning":   ["Advanced Machine Learning"],
    "deep learning":      ["Advanced Machine Learning"],
    "neural":             ["Advanced Machine Learning"],
    "ai":                 ["AI for Business Leaders", "Advanced Machine Learning"],
    "artificial intelligence": ["AI for Business Leaders", "Advanced Machine Learning"],
    "business":           ["AI for Business Leaders", "Fundamentals of Digital Marketing", "Project Management Fundamentals"],
    "blockchain":         ["Blockchain and Decentralized Applications"],
    "crypto":             ["Blockchain and Decentralized Applications"],
    "web3":               ["Blockchain and Decentralized Applications"],
    "cloud":              ["Cloud Computing Essentials"],
    "aws":                ["Cloud Computing Essentials"],
    "azure":              ["Cloud Computing Essentials"],
    "security":           ["Cybersecurity for Professionals", "Ethical Hacking Masterclass"],
    "cybersecurity":      ["Cybersecurity for Professionals"],
    "hacking":            ["Ethical Hacking Masterclass"],
    "ethical hacking":    ["Ethical Hacking Masterclass"],
    "data":               ["Data Visualization with Tableau", "Advanced Machine Learning"],
    "visualization":      ["Data Visualization with Tableau"],
    "tableau":            ["Data Visualization with Tableau"],
    "devops":             ["DevOps and Continuous Deployment"],
    "docker":             ["DevOps and Continuous Deployment"],
    "fitness":            ["Fitness and Nutrition Coaching"],
    "health":             ["Fitness and Nutrition Coaching"],
    "nutrition":          ["Fitness and Nutrition Coaching"],
    "marketing":          ["Fundamentals of Digital Marketing"],
    "seo":                ["Fundamentals of Digital Marketing"],
    "game":               ["Game Development with Unity"],
    "unity":              ["Game Development with Unity"],
    "gaming":             ["Game Development with Unity"],
    "design":             ["Graphic Design with Canva"],
    "graphic":            ["Graphic Design with Canva"],
    "canva":              ["Graphic Design with Canva"],
    "mobile":             ["Mobile App Development with Swift"],
    "ios":                ["Mobile App Development with Swift"],
    "swift":              ["Mobile App Development with Swift"],
    "networking":         ["Networking and System Administration"],
    "network":            ["Networking and System Administration"],
    "finance":            ["Personal Finance and Wealth Building", "Stock Market and Trading Strategies"],
    "money":              ["Personal Finance and Wealth Building", "Stock Market and Trading Strategies"],
    "investing":          ["Stock Market and Trading Strategies", "Personal Finance and Wealth Building"],
    "photography":        ["Photography and Video Editing"],
    "video":              ["Photography and Video Editing"],
    "editing":            ["Photography and Video Editing"],
    "project management": ["Project Management Fundamentals"],
    "agile":              ["Project Management Fundamentals"],
    "scrum":              ["Project Management Fundamentals"],
    "speaking":           ["Public Speaking Mastery"],
    "public speaking":    ["Public Speaking Mastery"],
    "communication":      ["Public Speaking Mastery"],
    "stock":              ["Stock Market and Trading Strategies"],
    "trading":            ["Stock Market and Trading Strategies"],
    "programming":        ["Python for Beginners", "Advanced Machine Learning", "Game Development with Unity", "Mobile App Development with Swift"],
    "coding":             ["Python for Beginners", "Advanced Machine Learning", "Game Development with Unity"],
}

def keyword_search(query: str, all_courses: list) -> list:
    q = query.strip().lower()
    if not q:
        return []
    seen, result = set(), []

    def add(courses):
        for c in courses:
            if c not in seen:
                seen.add(c); result.append(c)

    add([c for c in all_courses if q in c.lower()])
    for key in sorted(KEYWORD_MAP, key=len, reverse=True):
        if key in q or q in key:
            add(KEYWORD_MAP[key])
    tokens = re.findall(r"\w+", q)
    for c in all_courses:
        if any(t in c.lower() for t in tokens if len(t) > 2):
            add([c])
    return result


# ── Recommendation ─────────────────────────────────────────────────────────────
# Blend weights: topic similarity overrides nonsensical model cos-sim matches
TOPIC_W  = 0.50   # topic Jaccard
MODEL_W  = 0.20   # original cos_sim_df
COLLAB_W = 0.30   # KNN collaborative

def recommend(seed: str, top_n: int, m: dict) -> pd.Series:
    all_courses = m["all_courses"]

    def norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    topic  = norm(m["topic_sim"].loc[seed].reindex(all_courses, fill_value=0.0))
    model  = norm(m["cos_sim"].loc[seed].reindex(all_courses,   fill_value=0.0))

    # KNN collab on course×user matrix
    if seed in m["course_mat"].index:
        vec = m["course_mat"].loc[[seed]].values
        distances, indices = m["knn"].kneighbors(vec)
        collab_raw = pd.Series(1 - distances[0], index=[all_courses[i] for i in indices[0]])
        collab = norm(collab_raw.reindex(all_courses, fill_value=0.0))
    else:
        collab = pd.Series(0.0, index=all_courses)

    blended = TOPIC_W * topic + MODEL_W * model + COLLAB_W * collab
    blended = blended.drop(seed, errors="ignore")
    return blended.nlargest(top_n)


# ── UI helpers ─────────────────────────────────────────────────────────────────
ICONS = {
    "Python":("🐍","#d1fae5"), "Machine Learning":("🤖","#ede9fe"),
    "AI":("🧠","#dbeafe"), "Blockchain":("⛓️","#f3f4f6"),
    "Cloud":("☁️","#e0f2fe"), "Cybersecurity":("🔐","#fee2e2"),
    "Data":("📊","#fef9c3"), "DevOps":("⚙️","#f3f4f6"),
    "Ethical Hacking":("🛡️","#fee2e2"), "Fitness":("💪","#d1fae5"),
    "Marketing":("📣","#fef3c7"), "Game":("🎮","#ede9fe"),
    "Graphic":("🎨","#fce7f3"), "Mobile":("📱","#dbeafe"),
    "Networking":("🌐","#f3f4f6"), "Personal Finance":("💰","#d1fae5"),
    "Photography":("📷","#f3f4f6"), "Project":("📋","#fef3c7"),
    "Public Speaking":("🎤","#ede9fe"), "Stock":("📈","#d1fae5"),
    "Advanced":("🚀","#fef3c7"),
}

def get_icon(name):
    for kw, val in ICONS.items():
        if kw.lower() in name.lower():
            return val
    return ("📚","#f3f4f6")

def diff_pill(level):
    cls = {"Beginner":"pill-green","Intermediate":"pill-yellow","Advanced":"pill-red"}.get(level,"pill-blue")
    return f'<span class="pill {cls}">{level}</span>'

def stars_html(r):
    return "★" * int(r) + "☆" * (5 - int(r))

def render_card(rank, name, score, m):
    meta   = m["profiles"].loc[name].to_dict() if name in m["profiles"].index else {}
    stat   = m["stats"].loc[name].to_dict()    if name in m["stats"].index    else {}
    icon, bg = get_icon(name)
    diff   = meta.get("difficulty_level","")
    cert   = meta.get("certification_offered","No")
    inst   = meta.get("instructor","")
    rating = stat.get("rating",0)
    price  = stat.get("course_price",0)
    dur    = stat.get("course_duration_hours",0)
    enroll = int(stat.get("enrollment_numbers",0))
    pct    = int(score * 100)
    cert_h = '<span class="pill pill-purple">✓ Certificate</span>' if cert == "Yes" else ""

    st.markdown(f"""
    <div class="course-card">
      <div class="course-icon" style="background:{bg}">{icon}</div>
      <div class="course-body">
        <div class="course-title">#{rank} &nbsp; {name}</div>
        <div class="course-inst">👤 {inst}</div>
        <div class="course-meta">
          <span class="stars">{stars_html(rating)}</span>
          <strong>{rating:.1f}</strong>
          <span style="color:#d1d5db">|</span>
          {diff_pill(diff)} {cert_h}
          <span style="color:#d1d5db">|</span>
          ⏱ {dur:.0f}h &nbsp;·&nbsp; 👥 {enroll:,} &nbsp;·&nbsp; 💵 ${price:.0f}
        </div>
        <div class="match-bar-bg">
          <div class="match-bar-fg" style="width:{pct}%"></div>
        </div>
        <div class="match-label">Relevance score: {score:.3f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_seed_banner(seed, m):
    meta = m["profiles"].loc[seed].to_dict() if seed in m["profiles"].index else {}
    stat = m["stats"].loc[seed].to_dict()    if seed in m["stats"].index    else {}
    icon, bg = get_icon(seed)
    st.markdown(f"""
    <div style="background:rgba(124,58,237,0.1);border:1px solid rgba(167,139,250,0.2);
                border-radius:16px;padding:1rem 1.3rem;margin:1rem 0 1.5rem;
                display:flex;gap:1rem;align-items:center;">
      <div style="font-size:1.7rem;background:{bg};padding:0.45rem 0.55rem;
                  border-radius:12px;filter:saturate(0.8) brightness(0.9)">{icon}</div>
      <div>
        <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:3px">Showing recommendations based on</div>
        <div style="font-size:1rem;font-weight:700;color:#f1f5f9;font-family:'Space Grotesk',sans-serif">{seed}</div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.4);margin-top:3px">
          {meta.get('difficulty_level','')} &nbsp;·&nbsp;
          ⭐ {stat.get('rating',0):.1f} &nbsp;·&nbsp;
          ⏱ {stat.get('course_duration_hours',0):.0f}h
          {"&nbsp;·&nbsp; ✓ Certificate" if meta.get('certification_offered')=='Yes' else ""}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">🎓 CourseMatch</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(""):
        m = load_models()

    all_courses = m["all_courses"]

    st.markdown("""
    <div class="hero">
      <div class="hero-title">What do you want to <span>learn today?</span></div>
      <div class="hero-sub">Search by topic, skill, or keyword — we'll find the best courses for you</div>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("search", placeholder="🔍  Try 'machine learning', 'finance', 'python', 'design'…",
                              label_visibility="collapsed", key="search_query")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Search", type="primary", use_container_width=True)


    # Filters
    fc1, fc2, fc3 = st.columns([1, 2, 2])
    with fc1:
        top_n = st.selectbox("Show", [5, 8, 10])
    with fc2:
        diff_filter = st.multiselect("Difficulty", ["Beginner", "Intermediate", "Advanced"])
    with fc3:
        cert_filter = st.selectbox("Certificate", ["All", "With Certificate", "Without Certificate"])

    # ── Empty state ────────────────────────────────────────────────────────────
    if not query.strip():
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <div class="empty-title">Search for anything to get started</div>
          <div class="empty-sub">Try "machine learning", "investing", "photography", "python"</div>
        </div>
        """, unsafe_allow_html=True)
        return

    matched = keyword_search(query.strip(), all_courses)

    if not matched:
        st.markdown(f"""
        <div class="no-results">
          <div class="no-results-icon">😕</div>
          <div class="no-results-title">No courses found for "<strong>{query}</strong>"</div>
          <div class="no-results-sub">Try keywords like "python", "security", "finance", or "design"</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Seed selection
    st.markdown(f'<div class="section-title">Results for "<em>{query}</em>"</div>', unsafe_allow_html=True)

    if st.session_state.get("last_query") != query:
        st.session_state.selected_seed = matched[0]
        st.session_state.last_query    = query

    if len(matched) > 1:
        st.markdown("**Matched courses — click to see similar:**")
        pcols = st.columns(min(len(matched), 5))
        for i, course in enumerate(matched[:5]):
            label = ("🔵 " if course == st.session_state.get("selected_seed") else "") + course
            if pcols[i % 5].button(label, key=f"seed_{i}"):
                st.session_state.selected_seed = course

    seed = st.session_state.get("selected_seed", matched[0])
    render_seed_banner(seed, m)

    # Recommendations
    ranked = recommend(seed, top_n=config.TOP_N_MAX, m=m)

    # Apply filters
    filtered = []
    for name, score in ranked.items():
        meta = m["profiles"].loc[name].to_dict() if name in m["profiles"].index else {}
        if diff_filter and meta.get("difficulty_level") not in diff_filter:
            continue
        cert = meta.get("certification_offered", "No")
        if cert_filter == "With Certificate"    and cert != "Yes": continue
        if cert_filter == "Without Certificate" and cert == "Yes": continue
        filtered.append((name, score))
        if len(filtered) >= top_n:
            break

    if not filtered:
        st.info("No courses match your filters. Try adjusting difficulty or certificate options.")
        return

    st.markdown(f'<div class="section-sub">Showing {len(filtered)} courses similar to <strong>{seed}</strong></div>', unsafe_allow_html=True)

    for i, (name, score) in enumerate(filtered, 1):
        render_card(i, name, score, m)


if __name__ == "__main__":
    main()
