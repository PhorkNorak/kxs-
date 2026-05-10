"""
KhmerXScore LLM Scorers
=========================
Zero-shot scoring with GPT-4, Claude, and Gemini.
Captures both score and reasoning (for XAI Layer B comparison).
"""

import json
import re
import time
from typing import Dict, Optional
import numpy as np


# ============================================================
# Prompt template
# ============================================================
SCORING_PROMPT = """You are a Cambodian school teacher grading a student's short-answer response.

**Question:** {question}

**Reference Answer (full-score example):** {reference}

**Student's Answer:** {answer}

**Maximum Score:** {max_score}

**Instructions:**
1. Compare the student's answer to the reference answer.
2. Evaluate factual correctness, completeness, and relevance.
3. Note any spelling errors (these should reduce the score).
4. Assign a score from 0 to {max_score}.

**Respond in exactly this JSON format:**
{{
  "score": <integer 0 to {max_score}>,
  "reasoning": "<brief explanation in Khmer or English>"
}}
"""


def parse_llm_response(response_text: str, max_score: int) -> Dict:
    """Parse LLM response to extract score and reasoning."""
    try:
        # Try to parse as JSON
        # Handle markdown code blocks
        text = response_text.strip()
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        
        data = json.loads(text)
        score = int(data.get("score", 0))
        score = max(0, min(score, max_score))
        reasoning = str(data.get("reasoning", ""))
        
        return {"score": score, "reasoning": reasoning, "raw": response_text}
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to find a number
        numbers = re.findall(r"\b(\d+)\b", response_text)
        if numbers:
            score = int(numbers[0])
            score = max(0, min(score, max_score))
            return {"score": score, "reasoning": response_text, "raw": response_text}
        return {"score": 0, "reasoning": "PARSE_ERROR", "raw": response_text}


# ============================================================
# GPT-4 Scorer
# ============================================================
class GPT4Scorer:
    """Score using OpenAI GPT-4 API (zero-shot)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def score_one(self, question: str, reference: str, answer: str,
                  max_score: int) -> Dict:
        prompt = SCORING_PROMPT.format(
            question=question, reference=reference,
            answer=answer, max_score=max_score
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content
            return parse_llm_response(text, max_score)
        except Exception as e:
            return {"score": 0, "reasoning": f"API_ERROR: {e}", "raw": ""}
    
    def predict(self, df, delay: float = 0.5):
        return _predict_batch(self, df, delay)


# ============================================================
# Claude Scorer
# ============================================================
class ClaudeScorer:
    """Score using Anthropic Claude API (zero-shot)."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def score_one(self, question: str, reference: str, answer: str,
                  max_score: int) -> Dict:
        prompt = SCORING_PROMPT.format(
            question=question, reference=reference,
            answer=answer, max_score=max_score
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return parse_llm_response(text, max_score)
        except Exception as e:
            return {"score": 0, "reasoning": f"API_ERROR: {e}", "raw": ""}
    
    def predict(self, df, delay: float = 0.5):
        return _predict_batch(self, df, delay)


# ============================================================
# Gemini Scorer
# ============================================================
class GeminiScorer:
    """Score using Google Gemini API (zero-shot)."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def score_one(self, question: str, reference: str, answer: str,
                  max_score: int) -> Dict:
        prompt = SCORING_PROMPT.format(
            question=question, reference=reference,
            answer=answer, max_score=max_score
        )
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            return parse_llm_response(text, max_score)
        except Exception as e:
            return {"score": 0, "reasoning": f"API_ERROR: {e}", "raw": ""}
    
    def predict(self, df, delay: float = 0.5):
        return _predict_batch(self, df, delay)


# ============================================================
# Shared batch prediction
# ============================================================
def _predict_batch(scorer, df, delay: float = 0.5) -> Dict:
    """Run LLM scoring on entire DataFrame with rate limiting."""
    from tqdm import tqdm
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM scoring"):
        result = scorer.score_one(
            question=row["Question"],
            reference=row["Reference"],
            answer=row["Answer"],
            max_score=int(row["Max Score"]),
        )
        results.append(result)
        time.sleep(delay)
    
    # Compute normalized scores
    raw_scores = np.array([r["score"] for r in results])
    max_scores = df["Max Score"].values
    normalized = raw_scores / max_scores
    normalized = np.clip(normalized, 0, 1)
    labels = np.round(normalized * 4).astype(np.int64).clip(0, 4)
    
    return {
        "scores": normalized,
        "labels": labels,
        "raw_scores": raw_scores,
        "reasoning": [r["reasoning"] for r in results],
        "raw_responses": [r["raw"] for r in results],
    }


# ============================================================
# Factory
# ============================================================
def create_llm_scorer(name: str, api_key: str):
    """Create an LLM scorer by name."""
    scorers = {
        "gpt4": GPT4Scorer,
        "claude": ClaudeScorer,
        "gemini": GeminiScorer,
    }
    if name not in scorers:
        raise ValueError(f"Unknown LLM: {name}")
    return scorers[name](api_key=api_key)
