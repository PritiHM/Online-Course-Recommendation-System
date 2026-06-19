"""
config.py — App-wide settings for the Course Recommendation API
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models_sklearn_only.pkl")

# ── Server ─────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ── Recommendation defaults ────────────────────────────────────────────────────
# Number of nearest-neighbour courses to return (KNN was fitted with 10)
TOP_N_DEFAULT = int(os.getenv("TOP_N_DEFAULT", 5))
TOP_N_MAX     = 10          # hard ceiling — never exceed KNN n_neighbors

# Weight for blending content-based (cos-sim) vs. collaborative (SVD + KNN) scores
# 0.0 = pure collaborative  |  1.0 = pure content-based
CONTENT_WEIGHT = float(os.getenv("CONTENT_WEIGHT", 0.4))
COLLAB_WEIGHT  = 1.0 - CONTENT_WEIGHT

# ── Feature columns expected in preprocessed_df ────────────────────────────────
NUMERIC_FEATURES = [
    "course_duration_hours",
    "rating",
    "enrollment_numbers",
    "course_price",
    "feedback_score",
    "time_spent_hours",
    "previous_courses_taken",
]

CATEGORICAL_FEATURES = [
    "difficulty_level",
    "certification_offered",
    "study_material_available",
]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
