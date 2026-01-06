"""Recommendation utilities. Uses `ner.py` ranking when available, otherwise a simple fallback."""
from typing import List, Optional
import pandas as pd

try:
    import ner
    NER_AVAILABLE = True
except Exception:
    NER_AVAILABLE = False


def run_recommendations(df: pd.DataFrame, subject: str, top_n: int = 10) -> Optional[List[dict]]:
    """Return a list of top faculty for a subject.

    Uses `ner.rank_faculty` if available, otherwise falls back to `simple_recommendation`.
    """
    if df is None or df.empty:
        return None

    if NER_AVAILABLE:
        try:
            return ner.rank_faculty(df=df, subject=subject, top_n=top_n)
        except Exception:
            # fall back to simple method
            return simple_recommendation(df, subject, top_n=top_n)
    else:
        return simple_recommendation(df, subject, top_n=top_n)


def simple_recommendation(df: pd.DataFrame, subject: str, top_n: int = 10) -> Optional[List[dict]]:
    """Simple text-based matching fallback."""
    try:
        if df is None or df.empty:
            return None

        text_column = None
        cols_lower = [c.lower() for c in df.columns]

        for cand in ("bio", "description", "profile", "details", "about", "summary", "expertise", "field", "specialization", "skills"):
            for i, c in enumerate(cols_lower):
                if cand in c:
                    text_column = df.columns[i]
                    break
            if text_column:
                break

        name_column = None
        for cand in ("name", "faculty", "professor", "instructor", "staff"):
            for i, c in enumerate(cols_lower):
                if cand in c:
                    name_column = df.columns[i]
                    break
            if name_column:
                break

        if not text_column:
            text_column = df.columns[0]
        if not name_column:
            name_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        subject_lower = subject.lower()
        results = []

        for idx, row in df.iterrows():
            text = str(row.get(text_column, ""))
            name = str(row.get(name_column, f"Faculty {idx}"))
            text_lower = text.lower()

            score = text_lower.count(subject_lower) * 3
            subject_words = subject_lower.split()
            for word in subject_words:
                score += text_lower.count(word)

            age = row.get('Age', 'N/A')
            gender = row.get('Gender', 'N/A')
            experience = row.get('YearsExperience', row.get('Experience', 'N/A'))
            field = row.get('FieldOfExpertise', text[:100] + "...")

            results.append({
                "name": name,
                "score": float(score),
                "text": text,
                "index": idx,
                "age": age,
                "gender": gender,
                "years_experience": experience,
                "field_of_expertise": field
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_n]
    except Exception:
        return None
