"""
AI Telemetry Monitor & Autonomous Optimization Advisor.
Connects directly to Claude Sonnet 5 with Extended Thinking via Kaggle Model Proxy
to monitor training fold progress, analyze error residuals, and prescribe dynamic improvements.
"""

import os
import sys
import json
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

load_dotenv()

MODEL_PROXY_URL = os.environ.get("MODEL_PROXY_URL", "https://mp-staging.kaggle.net/models")
MODEL_PROXY_API_KEY = os.environ.get("MODEL_PROXY_API_KEY")
MODEL_ENDPOINT = f"{MODEL_PROXY_URL}/anthropic/claude-sonnet-5@default"


class LiveAITelemetryAdvisor:
    """
    Live AI Advisor that monitors cross-validation telemetry and prescribes improvements.
    """
    def __init__(self):
        self.api_key = MODEL_PROXY_API_KEY
        self.endpoint = MODEL_ENDPOINT

    def diagnose_and_prescribe(
        self,
        fold_scores: List[float],
        stacker_coefficients: Dict[str, float],
        model_scores: Dict[str, float],
        dataset_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sends live cross-validation metrics to Claude Sonnet 5 for real-time diagnostic synthesis.
        """
        mean_auc = sum(fold_scores) / len(fold_scores)
        std_auc = (sum((x - mean_auc) ** 2 for x in fold_scores) / len(fold_scores)) ** 0.5

        prompt = f"""
You are the Lead Decision Intelligence & ML Optimization Advisor for Kaggle Playground S6E8 (Predicting Smartphone Addiction).
The training run just completed on 691,369 rows with 10-fold Stratified CV.

Telemetry Metrics:
- 10-Fold OOF Mean AUC: {mean_auc:.5f} (Std Dev: {std_auc:.5f})
- Fold Scores: {[round(s, 5) for s in fold_scores]}
- Base Model OOF Scores: {json.dumps(model_scores, indent=2)}
- Stacker Meta-Coefficients: {json.dumps(stacker_coefficients, indent=2)}
- Dataset Metadata: {json.dumps(dataset_meta, indent=2)}

Please provide a concise, high-level mathematical diagnostic (in 3 structured bullet points):
1. **Fold Variance & Stability Audit**: Is the standard deviation ({std_auc:.5f}) within optimal bounds, or is there fold drift?
2. **Stacker Weight Analysis**: How are the negative/positive meta-weights behaving (e.g. error cancellation)?
3. **Prescriptive Action for Wave 6**: Recommend 1 concrete post-processing or architectural refinement to push AUC further.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "anthropic/claude-sonnet-5@default",
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            res = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            if res.status_code == 200:
                data = res.json()
                text_chunks = [b['text'] for b in data.get('content', []) if b.get('type') == 'text']
                advice = "\n".join(text_chunks)
                usage = data.get('usage', {})
                return {
                    "status": "success",
                    "advisor": "anthropic/claude-sonnet-5@default",
                    "advice": advice,
                    "tokens_used": usage
                }
            else:
                return {
                    "status": "error",
                    "error_code": res.status_code,
                    "message": res.text
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    advisor = LiveAITelemetryAdvisor()
    sample_scores = [0.9661, 0.9664, 0.9659, 0.9667, 0.9662, 0.9665, 0.9658, 0.9663, 0.9666, 0.9660]
    sample_coefs = {"LGB": 0.45, "XGB": 0.38, "CAT": 0.22, "NN": -0.05, "FM": 0.08}
    sample_models = {"LGB": 0.9648, "XGB": 0.9652, "CAT": 0.9639, "NN": 0.9580, "FM": 0.9592}
    meta = {"total_rows": 691369, "features": 31}

    print("🤖 Querying Live Claude Telemetry Advisor...")
    report = advisor.diagnose_and_prescribe(sample_scores, sample_coefs, sample_models, meta)
    print(json.dumps(report, indent=2))
