import re
from collections import Counter
import spacy


def load_nlp(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except Exception as e:
        raise RuntimeError(
            f"spaCy model '{model}' not available. Install it with: python -m spacy download {model}"
        ) from e


def rank_faculty(df, subject: str, text_column: str = None, name_column: str = None, top_n: int = 5, model: str = "en_core_web_sm"):
    """Rank faculty rows in `df` by relevance to `subject`.

    Expects `df` to contain a text column with faculty descriptions (bio/profile).
    The function will try to autodetect a name column and a text column if not provided.

    Returns a list of dicts: [{'name': ..., 'score': ..., 'text': ...}, ...]
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    cols_lower = [c.lower() for c in df.columns]

    # Autodetect name column
    if name_column is None:
        for cand in ("name", "faculty", "professor", "instructor", "staff"):
            for i, c in enumerate(cols_lower):
                if cand in c:
                    name_column = df.columns[i]
                    break
            if name_column:
                break

    # Autodetect text/bio column
    if text_column is None:
        # First check for FieldOfExpertise as it's likely the relevant text for faculty
        if 'FieldOfExpertise' in df.columns:
            text_column = 'FieldOfExpertise'
        else:
            for cand in ("bio", "description", "profile", "details", "about", "summary", "cv"):
                for i, c in enumerate(cols_lower):
                    if cand in c:
                        text_column = df.columns[i]
                        break
                if text_column:
                    break

    if text_column is None:
        # fallback to first non-name column or first column
        if name_column:
            text_column = next((c for c in df.columns if c != name_column), df.columns[0])
        else:
            text_column = df.columns[0]

    nlp = load_nlp(model)
    subj_doc = nlp(subject.lower())
    subj_tokens = [t.lemma_ for t in subj_doc if t.is_alpha and not t.is_stop]
    subj_set = set(subj_tokens)

    results = []
    for idx, row in df.iterrows():
        text = str(row.get(text_column, ""))
        text_l = text.lower()

        # direct substring counts for subject tokens
        count = sum(text_l.count(tok) for tok in subj_set) if subj_set else 0

        # token overlap using spaCy tokenization
        doc = nlp(text_l)
        doc_tokens = set([t.lemma_ for t in doc if t.is_alpha and not t.is_stop])
        overlap = len(subj_set & doc_tokens)

        # similarity (may be 0 for small models without vectors)
        sim = 0.0
        try:
            sim = subj_doc.similarity(doc)
        except Exception:
            sim = 0.0

        # simple scoring heuristic (tweakable)
        score = count * 3 + overlap * 2 + sim * 5

        if name_column and name_column in df.columns:
            name = row.get(name_column, str(idx))
        else:
            # try common name-like columns or fallback to index
            name = None
            for cand in ("name", "faculty", "professor"):
                for i, c in enumerate(cols_lower):
                    if cand in c:
                        name = row.get(df.columns[i], str(idx))
                        break
                if name:
                    break
            if not name:
                name = str(idx)

        results.append({
            "name": name,
            "score": float(score),
            "text": text,
            "index": idx,
            "age": row.get('Age', ''),
            "gender": row.get('Gender', ''),
            "years_experience": row.get('YearsExperience', ''),
            "field_of_expertise": text
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_n]
