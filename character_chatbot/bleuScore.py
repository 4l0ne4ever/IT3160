import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Hàm tính BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return score

# Hàm tính BLEU score trung bình và phân tích
def analyze_bleu_scores(filename):
    # Tải file JSON
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Tính BLEU score cho từng mẫu
    results = data["results"]
    bleu_scores = []
    false_bleu_scores = []
    
    for sample in results:
        bleu_score = calculate_bleu(sample["solid_response"], sample["chat_response"])
        sample["bleu_score"] = bleu_score
        bleu_scores.append(bleu_score)
        if not sample["accuracy"]:
            false_bleu_scores.append(bleu_score)
    
    # Tính trung bình và độ lệch chuẩn
    avg_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    avg_false_bleu = np.mean(false_bleu_scores) if false_bleu_scores else 0
    std_false_bleu = np.std(false_bleu_scores) if false_bleu_scores else 0
    
    # In kết quả
    print(f"\nAnalysis for {filename}:")
    print(f"Total samples: {len(results)}")
    print(f"Average BLEU score (all samples): {avg_bleu:.4f}")
    print(f"Std BLEU score (all samples): {std_bleu:.4f}")
    print(f"False samples: {len(false_bleu_scores)} ({len(false_bleu_scores)/len(results)*100:.2f}%)")
    print(f"Average BLEU score (false samples): {avg_false_bleu:.4f}")
    print(f"Std BLEU score (false samples): {std_false_bleu:.4f}")
    
    return bleu_scores, false_bleu_scores

# Tính cho cả hai file
bleu_scores_eval, false_bleu_scores_eval = analyze_bleu_scores("evaluation_results.json")
bleu_scores_qwen, false_bleu_scores_qwen = analyze_bleu_scores("evaluation_qwen_results.json")