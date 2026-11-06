"""End-to-End Decision Benchmark on Realistic Customer & Incident Queries.

Runs full ModernBERT + Decision Heads on real prompts to compare Base vs Improved weights:
- Gathers representations at [MASK] tokens
- Computes calibrated option logits & probabilities
- Evaluates Top-1 Choice, Margin, and Action/Deferral Confidence
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import ModernBertConfig, ModernBertModel

repo_root = Path(__file__).resolve().parent.parent

# Test cases representing enterprise scenarios
TEST_CASES = [
    {
        "id": "CASE-1-INFRA",
        "type": "choice",
        "question": "Which engineering team should handle this incident?",
        "options": ["billing", "infrastructure", "sales"],
        "state": "URGENT: Kubernetes production cluster nodes are in CrashLoopBackOff with 502 Bad Gateway across all regions.",
        "expected": "infrastructure",
    },
    {
        "id": "CASE-2-BILLING",
        "type": "choice",
        "question": "Which team should handle this customer inquiry?",
        "options": ["billing", "infrastructure", "sales"],
        "state": "The customer was charged twice on invoice INV-2024-9812 for the enterprise tier subscription.",
        "expected": "billing",
    },
    {
        "id": "CASE-3-SALES",
        "type": "choice",
        "question": "Which team should receive this inbound lead?",
        "options": ["billing", "infrastructure", "sales"],
        "state": "We are looking to upgrade from our free tier to an enterprise contract for 500 team seats.",
        "expected": "sales",
    },
    {
        "id": "CASE-4-SECURITY",
        "type": "choice",
        "question": "Which team should be alerted for this event?",
        "options": ["billing", "infrastructure", "sales"],
        "state": "Multiple failed SSH login attempts detected from unknown IP subnet targeting bastions.",
        "expected": "infrastructure",
    },
]


def load_full_pipeline(checkpoint_path: str):
    cfg = ModernBertConfig.from_pretrained(str(repo_root / "models" / "rev-decision-model" / "encoder" / "config.json"))
    encoder = ModernBertModel(cfg)
    sf = safe_open(checkpoint_path, framework="pt")

    enc_sd = {k.replace("encoder.", ""): sf.get_tensor(k) for k in sf.keys() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd, strict=False)
    encoder.eval()

    tok = Tokenizer.from_file(str(repo_root / "models" / "rev-decision-model" / "tokenizer" / "tokenizer.json"))

    # Load head weights
    type_emb = sf.get_tensor("type_emb.weight").float()
    scorer_norm_w = sf.get_tensor("scorer.0.weight").float()
    scorer_norm_b = sf.get_tensor("scorer.0.bias").float()
    scorer_1_w = sf.get_tensor("scorer.1.weight").float()
    scorer_1_b = sf.get_tensor("scorer.1.bias").float()
    scorer_3_w = sf.get_tensor("scorer.3.weight").float()
    scorer_3_b = sf.get_tensor("scorer.3.bias").float()

    act_0_w = sf.get_tensor("act_head.0.weight").float()
    act_0_b = sf.get_tensor("act_head.0.bias").float()
    act_2_w = sf.get_tensor("act_head.2.weight").float()
    act_2_b = sf.get_tensor("act_head.2.bias").float()

    return {
        "encoder": encoder,
        "tokenizer": tok,
        "type_emb": type_emb,
        "scorer_norm": (scorer_norm_w, scorer_norm_b),
        "scorer_1": (scorer_1_w, scorer_1_b),
        "scorer_3": (scorer_3_w, scorer_3_b),
        "act_0": (act_0_w, act_0_b),
        "act_2": (act_2_w, act_2_b),
    }


def predict(pipeline, test_case):
    tok = pipeline["tokenizer"]
    options = test_case["options"]
    opt_str = " ".join([f"[MASK] {o}" for o in options])
    prompt = f"[CLS] choice question: {test_case['question']} [SEP] {opt_str} [SEP] {test_case['state']} [SEP]"

    enc = tok.encode(prompt)
    input_ids = torch.tensor([enc.ids])

    # Find [MASK] token positions (50284 is [MASK])
    mask_indices = [i for i, token_id in enumerate(enc.ids) if tok.id_to_token(token_id) == "[MASK]"]

    with torch.no_grad():
        out = pipeline["encoder"](input_ids)
        hidden = out.last_hidden_state[0]  # [seq_len, 1024]

        # Gather at mask tokens
        gathered = hidden[mask_indices]  # [N, 1024]
        h = gathered + pipeline["type_emb"][0]  # choice type

        # Scorer
        norm_w, norm_b = pipeline["scorer_norm"]
        h_norm = nn.functional.layer_norm(h, (1024,), norm_w, norm_b)

        s1_w, s1_b = pipeline["scorer_1"]
        h1 = nn.functional.gelu(h_norm @ s1_w.t() + s1_b)

        s3_w, s3_b = pipeline["scorer_3"]
        logits = (h1 @ s3_w.t() + s3_b).squeeze(-1)  # [N]
        probs = torch.softmax(logits / 1.0346, dim=-1)

        # Action head
        pooled = gathered.mean(dim=0, keepdim=True)
        sorted_p, _ = torch.sort(probs, descending=True)
        margin = (sorted_p[0] - sorted_p[1]).unsqueeze(0).unsqueeze(1)
        entropy = (-torch.sum(probs * torch.log(probs + 1e-9))).unsqueeze(0).unsqueeze(1)
        max_p = sorted_p[0:1].unsqueeze(1)
        std_p = probs.std().unsqueeze(0).unsqueeze(1)

        feat = torch.cat([pooled, margin, entropy, max_p, std_p], dim=-1)
        a0_w, a0_b = pipeline["act_0"]
        h_act = nn.functional.gelu(feat @ a0_w.t() + a0_b)
        a2_w, a2_b = pipeline["act_2"]
        act_logits = (h_act @ a2_w.t() + a2_b)[0]
        act_probs = torch.softmax(act_logits, dim=-1)

    top_idx = int(probs.argmax().item())
    pred_option = options[top_idx]
    confidence = float(probs[top_idx].item())
    top_margin = float(margin.item())
    defer_prob = float(act_probs[0].item())
    act_prob = float(act_probs[1].item())

    return {
        "predicted": pred_option,
        "is_correct": pred_option == test_case["expected"],
        "confidence": confidence,
        "margin": top_margin,
        "defer_prob": defer_prob,
        "act_prob": act_prob,
        "decision": "ACT" if act_prob > defer_prob else "DEFER",
    }


def main():
    print("=" * 70)
    print("END-TO-END BENCHMARK: Base Model vs WSP Improved Model")
    print("=" * 70)

    base_p = str(repo_root / "models" / "rev-decision-model" / "model.safetensors")
    imp_p = str(repo_root / "models" / "rev-decision-model" / "rev_decision_improved.safetensors")

    print("\n--- Testing Base Model Checkpoint ---")
    base_pipe = load_full_pipeline(base_p)
    base_results = []
    for tc in TEST_CASES:
        res = predict(base_pipe, tc)
        base_results.append(res)
        print(f"[{tc['id']}] Pred: {res['predicted']} (Target: {tc['expected']}) | Conf: {res['confidence']:.3f} | Margin: {res['margin']:+.3f} | Decision: {res['decision']} (Act={res['act_prob']:.2f}, Defer={res['defer_prob']:.2f})")

    print("\n--- Testing WSP Improved Model Checkpoint ---")
    imp_pipe = load_full_pipeline(imp_p)
    imp_results = []
    for tc in TEST_CASES:
        res = predict(imp_pipe, tc)
        imp_results.append(res)
        print(f"[{tc['id']}] Pred: {res['predicted']} (Target: {tc['expected']}) | Conf: {res['confidence']:.3f} | Margin: {res['margin']:+.3f} | Decision: {res['decision']} (Act={res['act_prob']:.2f}, Defer={res['defer_prob']:.2f})")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    base_acc = sum(r["is_correct"] for r in base_results) / len(base_results)
    imp_acc = sum(r["is_correct"] for r in imp_results) / len(imp_results)
    base_avg_margin = sum(r["margin"] for r in base_results) / len(base_results)
    imp_avg_margin = sum(r["margin"] for r in imp_results) / len(imp_results)
    base_act_rate = sum(r["decision"] == "ACT" for r in base_results) / len(base_results)
    imp_act_rate = sum(r["decision"] == "ACT" for r in imp_results) / len(imp_results)

    print(f"Accuracy:           Base = {base_acc*100:.1f}% -> Improved = {imp_acc*100:.1f}%")
    print(f"Average Margin:     Base = {base_avg_margin:.4f} -> Improved = {imp_avg_margin:.4f} (Delta: {imp_avg_margin - base_avg_margin:+.4f})")
    print(f"Autonomous Action:  Base = {base_act_rate*100:.1f}% -> Improved = {imp_act_rate*100:.1f}%")

    out_json = repo_root / "models" / "rev-decision-model" / "end_to_end_benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "base_results": base_results,
            "improved_results": imp_results,
            "accuracy": {"base": base_acc, "improved": imp_acc},
            "average_margin": {"base": base_avg_margin, "improved": imp_avg_margin},
            "autonomous_action_rate": {"base": base_act_rate, "improved": imp_act_rate},
        }, f, indent=2)
    print(f"\nReport written to: {out_json}")


if __name__ == "__main__":
    main()
