import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
from rouge_score import rouge_scorer


# ========================
# 1. Khởi tạo và đọc dữ liệu
# ========================


# Kiểm tra và chọn device phù hợp (GPU NVIDIA hoặc CPU)
# Nếu có GPU NVIDIA sẽ dùng để tăng tốc, nếu không sẽ dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Đọc dữ liệu test từ hai file JSON
# Mỗi file chứa các cặp câu hỏi-câu trả lời từ mỗi mô hình
with open("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/character_chatbot/result/qwen_results.json", "r") as f:
   qwen_data = json.load(f)
with open("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/character_chatbot/result/chatbot_results.json", "r") as f:
   llama_data = json.load(f)


# ========================
# 2. Cấu hình mô hình reranker để đánh giá ngữ nghĩa
# ========================
# Sử dụng mô hình Alibaba-NLP/gte-multilingual-reranker-base để tính điểm tương đồng ngữ nghĩa
model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
   model_name,
   trust_remote_code=True,
   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
   device_map="auto"
)
model.eval()
model.to(device)


# Khởi tạo scorer ROUGE-L (chỉ cần tạo 1 lần)
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# ========================
# 3. Định nghĩa các hàm tính toán các chỉ số đánh giá
# ========================


def calculate_response_length(response):
   """
   Tính số từ trong câu trả lời.
   Args:
       response (str): Câu trả lời
   Returns:
       int: Số từ
   """
   return len(response.split())


def calculate_word_overlap(response1, response2):
   """
   Tính tỷ lệ từ trùng lặp giữa hai câu trả lời.
   Args:
       response1 (str): Câu trả lời thứ nhất
       response2 (str): Câu trả lời thứ hai
   Returns:
       float: Tỷ lệ từ trùng lặp (0-1)
   """
   words1 = set(response1.lower().split())
   words2 = set(response2.lower().split())
   overlap = words1.intersection(words2)
   return len(overlap) / max(len(words1), len(words2))


def calculate_bleu(reference, hypothesis):
   """
   Tính điểm BLEU đơn giản hóa giữa câu tham chiếu và câu dự đoán.
   BLEU ở đây chỉ tính precision và brevity penalty, không tính n-gram.
   Args:
       reference (str): Câu tham chiếu (chuẩn)
       hypothesis (str): Câu dự đoán (từ model)
   Returns:
       float: Điểm BLEU (0-1)
   """
   reference_tokens = [reference.lower().split()]
   hypothesis_tokens = hypothesis.lower().split()
   if len(hypothesis_tokens) == 0 or len(reference_tokens[0]) == 0:
       return 0.0
   matches = sum(1 for w in hypothesis_tokens if w in reference_tokens[0])
   precision = matches / len(hypothesis_tokens)
   brevity_penalty = min(1.0, len(hypothesis_tokens) / len(reference_tokens[0]))
   return precision * brevity_penalty


def calculate_vocabulary_diversity(text):
   """
   Tính độ đa dạng từ vựng (type-token ratio).
   Args:
       text (str): Văn bản cần đánh giá
   Returns:
       float: Tỷ lệ từ vựng đa dạng (0-1)
   """
   words = text.lower().split()
   if not words:
       return 0.0
   unique_words = set(words)
   return len(unique_words) / len(words)


def calculate_repetition_rate(text):
   """
   Tính tỷ lệ lặp lại từ trong câu trả lời.
   Args:
       text (str): Văn bản cần đánh giá
   Returns:
       float: Tỷ lệ lặp lại (0-1), càng thấp càng tự nhiên
   """
   words = text.lower().split()
   if not words:
       return 0.0
   word_counts = {}
   for word in words:
       word_counts[word] = word_counts.get(word, 0) + 1
   total_repetitions = sum(count - 1 for count in word_counts.values() if count > 1)
   return total_repetitions / len(words)


def calculate_rouge_l(reference, hypothesis):
   """
   Tính điểm ROUGE-L giữa câu tham chiếu và câu dự đoán.
   Args:
       reference (str): Câu tham chiếu (chuẩn)
       hypothesis (str): Câu dự đoán (từ model)
   Returns:
       float: ROUGE-L F1 score (0-1)
   """
   scores = rouge_scorer_obj.score(reference, hypothesis)
   return scores['rougeL'].fmeasure


# ========================
# 4. Hàm tính điểm ngữ nghĩa bằng reranker
# ========================
def score_pairs(pairs, tokenizer, model, device):
   """
   Tính điểm tương đồng ngữ nghĩa giữa các cặp câu bằng reranker.
   Args:
       pairs (list): Danh sách các cặp câu [ [message, response], ... ]
       tokenizer: Tokenizer của mô hình reranker
       model: Mô hình reranker
       device: CPU/GPU
   Returns:
       numpy.ndarray: Mảng điểm số đã chuẩn hóa về [0,1]
   """
   with torch.no_grad():
       inputs = tokenizer(
           pairs,
           padding=True,
           truncation=True,
           return_tensors="pt",
           max_length=512
       ).to(device)
       scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
       normalized_scores = (scores + 1) / 2
   return normalized_scores.cpu().numpy()


# ========================
# 5. Hàm vẽ biểu đồ so sánh các chỉ số
# ========================
def plot_scores_comparison(qwen_scores, llama_scores, solid_scores, result_dir):
   """
   Vẽ biểu đồ so sánh điểm ngữ nghĩa giữa các mô hình và câu trả lời chuẩn.
   Args:
       qwen_scores (list): Điểm Qwen
       llama_scores (list): Điểm Llama
       solid_scores (list): Điểm chuẩn
       result_dir (str): Thư mục lưu biểu đồ
   """
   indices = list(range(1, len(qwen_scores) + 1))
   plt.figure(figsize=(12, 6))
   plt.plot(indices, solid_scores, marker='o', label='Điểm chuẩn', color='green')
   plt.plot(indices, qwen_scores, marker='x', label='Qwen', color='blue')
   plt.plot(indices, llama_scores, marker='s', label='Llama', color='red')
   plt.xlabel('Cặp câu trả lời')
   plt.ylabel('Điểm số')
   plt.title('So sánh điểm số giữa các mô hình')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   plt.savefig(f"{result_dir}/scores_comparison.png")
   plt.close()


def plot_performance_distribution(qwen_dist, llama_dist, result_dir):
   """
   Vẽ biểu đồ phân bố chất lượng câu trả lời theo các mức: xuất sắc, tốt, trung bình, kém.
   Args:
       qwen_dist (dict): Phân bố Qwen
       llama_dist (dict): Phân bố Llama
       result_dir (str): Thư mục lưu biểu đồ
   """
   categories = ['Xuất sắc', 'Tốt', 'Trung bình', 'Kém']
   qwen_values = [qwen_dist['excellent'], qwen_dist['good'], qwen_dist['fair'], qwen_dist['poor']]
   llama_values = [llama_dist['excellent'], llama_dist['good'], llama_dist['fair'], llama_dist['poor']]
   x = np.arange(len(categories))
   width = 0.35
   plt.figure(figsize=(10, 6))
   plt.bar(x - width/2, qwen_values, width, label='Qwen', color='blue')
   plt.bar(x + width/2, llama_values, width, label='Llama', color='red')
   plt.xlabel('Chất lượng')
   plt.ylabel('Số lượng mẫu')
   plt.title('Phân bố chất lượng câu trả lời')
   plt.xticks(x, categories)
   plt.legend()
   plt.grid(True, axis='y')
   plt.tight_layout()
   plt.savefig(f"{result_dir}/performance_distribution.png")
   plt.close()


def plot_metrics_comparison(qwen_metrics, llama_metrics, result_dir):
   """
   Vẽ biểu đồ so sánh các metrics tổng hợp giữa hai mô hình.
   Args:
       qwen_metrics (dict): Metrics Qwen
       llama_metrics (dict): Metrics Llama
       result_dir (str): Thư mục lưu biểu đồ
   """
   metrics = ['Độ chính xác', 'Độ chênh lệch độ dài', 'Độ trùng lặp từ', 'Độ dài trung bình', 'Đa dạng từ vựng', 'Tỷ lệ lặp từ']
   qwen_values = [
       qwen_metrics['accuracy'],
       qwen_metrics['avg_length_diff'],
       qwen_metrics['avg_word_overlap'],
       qwen_metrics['avg_response_length'],
       qwen_metrics['avg_vocab_diversity'],
       qwen_metrics['avg_repetition_rate']
   ]
   llama_values = [
       llama_metrics['accuracy'],
       llama_metrics['avg_length_diff'],
       llama_metrics['avg_word_overlap'],
       llama_metrics['avg_response_length'],
       llama_metrics['avg_vocab_diversity'],
       llama_metrics['avg_repetition_rate']
   ]
   x = np.arange(len(metrics))
   width = 0.35
   plt.figure(figsize=(12, 6))
   plt.bar(x - width/2, qwen_values, width, label='Qwen', color='blue')
   plt.bar(x + width/2, llama_values, width, label='Llama', color='red')
   plt.xlabel('Metrics')
   plt.ylabel('Giá trị')
   plt.title('So sánh các metrics')
   plt.xticks(x, metrics, rotation=45)
   plt.legend()
   plt.grid(True, axis='y')
   plt.tight_layout()
   plt.savefig(f"{result_dir}/metrics_comparison.png")
   plt.close()


# ========================
# 6. Hàm đánh giá mô hình và tổng hợp kết quả
# ========================
def evaluate_model(data, model_name):
   """
   Đánh giá một mô hình chatbot trên tập dữ liệu.
   Args:
       data (list): Danh sách các cặp câu hỏi-câu trả lời
       model_name (str): Tên mô hình
   Returns:
       dict: Kết quả đánh giá chi tiết gồm các trường:
           - model_name: Tên mô hình
           - total_samples: Số mẫu
           - accuracy: Tỷ lệ mẫu trả lời đúng (theo ngưỡng điểm ngữ nghĩa)
           - avg_length_diff: Độ chênh lệch độ dài trung bình
           - avg_word_overlap: Độ trùng lặp từ trung bình
           - avg_bleu: BLEU trung bình
           - avg_rouge_l: ROUGE-L trung bình
           - avg_response_length: Độ dài câu trả lời trung bình
           - avg_vocab_diversity: Đa dạng từ vựng trung bình
           - avg_repetition_rate: Tỷ lệ lặp lại từ trung bình
           - avg_semantic_score: Điểm ngữ nghĩa trung bình của mô hình
           - performance_distribution: Phân bố chất lượng (xuất sắc, tốt, trung bình, kém)
           - results: Danh sách kết quả chi tiết từng mẫu
   """
   correct = 0
   threshold = 0.15  # Ngưỡng chấp nhận độ chênh lệch điểm ngữ nghĩa
   results = []
   total_length_diff = 0
   total_word_overlap = 0
   total_bleu = 0
   total_rouge_l = 0
   total_response_length = 0
   total_vocab_diversity = 0
   total_repetition_rate = 0
   total_semantic_score = 0


   for sample in data:
       message = sample["message"]
       chat_response = sample["chat_response"]
       solid_response = sample["solid_response"]
       # Tính điểm ngữ nghĩa giữa message và chat_response, message và solid_response
       pairs = [[message, chat_response], [message, solid_response]]
       scores = score_pairs(pairs, tokenizer, model, device)
       # Tính các metrics
       length_diff = abs(calculate_response_length(chat_response) - calculate_response_length(solid_response))
       word_overlap = calculate_word_overlap(chat_response, solid_response)
       bleu_score = calculate_bleu(solid_response, chat_response)
       rouge_l_score = calculate_rouge_l(solid_response, chat_response)
       response_length = calculate_response_length(chat_response)
       vocab_diversity = calculate_vocabulary_diversity(chat_response)
       repetition_rate = calculate_repetition_rate(chat_response)
       semantic_score = float(scores[0])
       # Cộng dồn để tính trung bình
       total_length_diff += length_diff
       total_word_overlap += word_overlap
       total_bleu += bleu_score
       total_rouge_l += rouge_l_score
       total_response_length += response_length
       total_vocab_diversity += vocab_diversity
       total_repetition_rate += repetition_rate
       total_semantic_score += semantic_score
       # Đánh giá đúng/sai dựa trên điểm ngữ nghĩa
       is_correct = abs(scores[0] - scores[1]) <= threshold or scores[0] >= scores[1]
       correct += 1 if is_correct else 0
       # Lưu kết quả chi tiết từng mẫu
       results.append({
           "message": message,  # Câu hỏi
           "chat_response": chat_response,  # Câu trả lời của mô hình
           "solid_response": solid_response,  # Câu trả lời chuẩn
           "semantic_score": semantic_score,  # Điểm ngữ nghĩa giữa message và chat_response
           "solid_score": float(scores[1]),    # Điểm ngữ nghĩa giữa message và solid_response
           "length_diff": length_diff,         # Độ chênh lệch độ dài
           "word_overlap": word_overlap,       # Độ trùng lặp từ
           "bleu": bleu_score,                 # BLEU
           "rouge_l": rouge_l_score,           # ROUGE-L
           "is_correct": bool(is_correct),     # Đúng/sai
           "response_length": response_length, # Số từ câu trả lời
           "vocab_diversity": vocab_diversity, # Đa dạng từ vựng
           "repetition_rate": repetition_rate  # Tỷ lệ lặp lại từ
       })
       if device.type == "cuda":
           torch.cuda.empty_cache()
   # Tính các metrics trung bình
   accuracy = correct / len(data)
   avg_length_diff = total_length_diff / len(data)
   avg_word_overlap = total_word_overlap / len(data)
   avg_bleu = total_bleu / len(data)
   avg_rouge_l = total_rouge_l / len(data)
   avg_response_length = total_response_length / len(data)
   avg_vocab_diversity = total_vocab_diversity / len(data)
   avg_repetition_rate = total_repetition_rate / len(data)
   avg_semantic_score = total_semantic_score / len(data)
   return {
       "model_name": model_name,
       "total_samples": len(data),
       "accuracy": accuracy,
       "avg_length_diff": avg_length_diff,
       "avg_word_overlap": avg_word_overlap,
       "avg_bleu": avg_bleu,
       "avg_rouge_l": avg_rouge_l,
       "avg_response_length": avg_response_length,
       "avg_vocab_diversity": avg_vocab_diversity,
       "avg_repetition_rate": avg_repetition_rate,
       "avg_semantic_score": avg_semantic_score,
       "performance_distribution": {
           "excellent": len([r for r in results if r["semantic_score"] >= 0.8]),
           "good": len([r for r in results if 0.6 <= r["semantic_score"] < 0.8]),
           "fair": len([r for r in results if 0.4 <= r["semantic_score"] < 0.6]),
           "poor": len([r for r in results if r["semantic_score"] < 0.4])
       },
       "results": results
   }


# ========================
# 7. Xử lý kết quả, lưu file, in báo cáo
# ========================
# Đánh giá cả hai mô hình
qwen_eval = evaluate_model(qwen_data, "Qwen/Qwen3-4B")
llama_eval = evaluate_model(llama_data, "eta-llama/Llama-3.2-3B-Instruct")


# Tính chênh lệch BLEU
bleu_diff = abs(qwen_eval["avg_bleu"] - llama_eval["avg_bleu"])


# Tạo thư mục kết quả với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = f"evaluation_results_{timestamp}"
os.makedirs(result_dir, exist_ok=True)


# Tạo báo cáo so sánh
comparison_report = {
   "evaluation_time": timestamp,
   "models": {
       "qwen": {
           "name": qwen_eval["model_name"],
           "metrics": {
               "accuracy": round(qwen_eval["accuracy"], 4),
               "average_length_difference": round(qwen_eval["avg_length_diff"], 2),
               "average_word_overlap": round(qwen_eval["avg_word_overlap"], 4),
               "average_bleu": round(qwen_eval["avg_bleu"], 4),
               "average_rouge_l": round(qwen_eval["avg_rouge_l"], 4),
               "average_response_length": round(qwen_eval["avg_response_length"], 2),
               "average_vocabulary_diversity": round(qwen_eval["avg_vocab_diversity"], 4),
               "average_repetition_rate": round(qwen_eval["avg_repetition_rate"], 4),
               "average_semantic_score": round(qwen_eval["avg_semantic_score"], 4)
           },
           "performance_distribution": qwen_eval["performance_distribution"]
       },
       "llama": {
           "name": llama_eval["model_name"],
           "metrics": {
               "accuracy": round(llama_eval["accuracy"], 4),
               "average_length_difference": round(llama_eval["avg_length_diff"], 2),
               "average_word_overlap": round(llama_eval["avg_word_overlap"], 4),
               "average_bleu": round(llama_eval["avg_bleu"], 4),
               "average_rouge_l": round(llama_eval["avg_rouge_l"], 4),
               "average_response_length": round(llama_eval["avg_response_length"], 2),
               "average_vocabulary_diversity": round(llama_eval["avg_vocab_diversity"], 4),
               "average_repetition_rate": round(llama_eval["avg_repetition_rate"], 4),
               "average_semantic_score": round(llama_eval["avg_semantic_score"], 4)
           },
           "performance_distribution": llama_eval["performance_distribution"]
       }
   },
   "bleu_difference": round(bleu_diff, 4)
}


# Lưu báo cáo so sánh
with open(f"{result_dir}/model_comparison.json", "w") as f:
   json.dump(comparison_report, f, ensure_ascii=False, indent=2)


# Tạo và lưu file kết quả chi tiết cho từng mẫu
detailed_results = []
for i in range(len(qwen_data)):
   detailed_result = {
       "sample_id": i + 1,
       "message": qwen_data[i]["message"],
       "solid_response": qwen_data[i]["solid_response"],
       "qwen": {
           "chat_response": qwen_eval["results"][i]["chat_response"],
           "semantic_score": round(qwen_eval["results"][i]["semantic_score"], 4),
           "length_diff": qwen_eval["results"][i]["length_diff"],
           "word_overlap": round(qwen_eval["results"][i]["word_overlap"], 4),
           "bleu_score": round(qwen_eval["results"][i]["bleu"], 4),
           "rouge_l": round(qwen_eval["results"][i]["rouge_l"], 4),
           "is_correct": qwen_eval["results"][i]["is_correct"],
           "response_length": qwen_eval["results"][i]["response_length"],
           "vocab_diversity": round(qwen_eval["results"][i]["vocab_diversity"], 4),
           "repetition_rate": round(qwen_eval["results"][i]["repetition_rate"], 4)
       },
       "llama": {
           "chat_response": llama_eval["results"][i]["chat_response"],
           "semantic_score": round(llama_eval["results"][i]["semantic_score"], 4),
           "length_diff": llama_eval["results"][i]["length_diff"],
           "word_overlap": round(llama_eval["results"][i]["word_overlap"], 4),
           "bleu_score": round(llama_eval["results"][i]["bleu"], 4),
           "rouge_l": round(llama_eval["results"][i]["rouge_l"], 4),
           "is_correct": llama_eval["results"][i]["is_correct"],
           "response_length": llama_eval["results"][i]["response_length"],
           "vocab_diversity": round(llama_eval["results"][i]["vocab_diversity"], 4),
           "repetition_rate": round(llama_eval["results"][i]["repetition_rate"], 4)
       },
       "comparison": {
           "semantic_score_diff": round(abs(qwen_eval["results"][i]["semantic_score"] - llama_eval["results"][i]["semantic_score"]), 4),
           "length_diff_diff": abs(qwen_eval["results"][i]["length_diff"] - llama_eval["results"][i]["length_diff"]),
           "word_overlap_diff": round(abs(qwen_eval["results"][i]["word_overlap"] - llama_eval["results"][i]["word_overlap"]), 4),
           "bleu_score_diff": round(abs(qwen_eval["results"][i]["bleu"] - llama_eval["results"][i]["bleu"]), 4),
           "rouge_l_diff": round(abs(qwen_eval["results"][i]["rouge_l"] - llama_eval["results"][i]["rouge_l"]), 4),
           "response_length_diff": abs(qwen_eval["results"][i]["response_length"] - llama_eval["results"][i]["response_length"]),
           "vocab_diversity_diff": round(abs(qwen_eval["results"][i]["vocab_diversity"] - llama_eval["results"][i]["vocab_diversity"]), 4),
           "repetition_rate_diff": round(abs(qwen_eval["results"][i]["repetition_rate"] - llama_eval["results"][i]["repetition_rate"]), 4)
       }
   }
   detailed_results.append(detailed_result)


# Lưu kết quả chi tiết vào file JSON
with open(f"{result_dir}/detailed_results.json", "w") as f:
   json.dump({
       "evaluation_time": timestamp,
       "total_samples": len(detailed_results),
       "overall_metrics": {
           "qwen": {
               "accuracy": round(qwen_eval["accuracy"], 4),
               "avg_length_diff": round(qwen_eval["avg_length_diff"], 2),
               "avg_word_overlap": round(qwen_eval["avg_word_overlap"], 4),
               "avg_bleu": round(qwen_eval["avg_bleu"], 4),
               "avg_rouge_l": round(qwen_eval["avg_rouge_l"], 4),
               "avg_response_length": round(qwen_eval["avg_response_length"], 2),
               "avg_vocab_diversity": round(qwen_eval["avg_vocab_diversity"], 4),
               "avg_repetition_rate": round(qwen_eval["avg_repetition_rate"], 4),
               "avg_semantic_score": round(qwen_eval["avg_semantic_score"], 4)
           },
           "llama": {
               "accuracy": round(llama_eval["accuracy"], 4),
               "avg_length_diff": round(llama_eval["avg_length_diff"], 2),
               "avg_word_overlap": round(llama_eval["avg_word_overlap"], 4),
               "avg_bleu": round(llama_eval["avg_bleu"], 4),
               "avg_rouge_l": round(llama_eval["avg_rouge_l"], 4),
               "avg_response_length": round(llama_eval["avg_response_length"], 2),
               "avg_vocab_diversity": round(llama_eval["avg_vocab_diversity"], 4),
               "avg_repetition_rate": round(llama_eval["avg_repetition_rate"], 4),
               "avg_semantic_score": round(llama_eval["avg_semantic_score"], 4)
           }
       },
       "samples": detailed_results
   }, f, ensure_ascii=False, indent=2)


# Vẽ các biểu đồ so sánh
qwen_scores = [r["semantic_score"] for r in qwen_eval["results"]]
llama_scores = [r["semantic_score"] for r in llama_eval["results"]]
solid_scores = [r["solid_score"] for r in qwen_eval["results"]]


plot_scores_comparison(qwen_scores, llama_scores, solid_scores, result_dir)
plot_performance_distribution(qwen_eval["performance_distribution"], llama_eval["performance_distribution"], result_dir)
plot_metrics_comparison(
   {
       "accuracy": qwen_eval["accuracy"],
       "avg_length_diff": qwen_eval["avg_length_diff"],
       "avg_word_overlap": qwen_eval["avg_word_overlap"],
       "avg_response_length": qwen_eval["avg_response_length"],
       "avg_vocab_diversity": qwen_eval["avg_vocab_diversity"],
       "avg_repetition_rate": qwen_eval["avg_repetition_rate"]
   },
   {
       "accuracy": llama_eval["accuracy"],
       "avg_length_diff": llama_eval["avg_length_diff"],
       "avg_word_overlap": llama_eval["avg_word_overlap"],
       "avg_response_length": qwen_eval["avg_response_length"],
       "avg_vocab_diversity": qwen_eval["avg_vocab_diversity"],
       "avg_repetition_rate": qwen_eval["avg_repetition_rate"]
   },
   result_dir
)


# In kết quả so sánh
print(f"\n=== Kết quả so sánh hai mô hình ({timestamp}) ===")
print("\n1. Qwen/Qwen3-4B:")
print(f"Độ chính xác: {qwen_eval['accuracy']:.2%}")
print(f"Độ chênh lệch độ dài trung bình: {qwen_eval['avg_length_diff']:.2f} từ")
print(f"Độ trùng lặp từ trung bình: {qwen_eval['avg_word_overlap']:.2%}")
print(f"BLEU trung bình: {qwen_eval['avg_bleu']:.4f}")
print(f"Độ dài câu trả lời trung bình: {qwen_eval['avg_response_length']:.2f} từ")
print(f"Đa dạng từ vựng trung bình: {qwen_eval['avg_vocab_diversity']:.4f}")
print(f"Tỷ lệ lặp lại từ trung bình: {qwen_eval['avg_repetition_rate']:.4f}")
print("\nPhân bố chất lượng:")
print(f"- Xuất sắc (≥0.8): {qwen_eval['performance_distribution']['excellent']} mẫu")
print(f"- Tốt (0.6-0.8): {qwen_eval['performance_distribution']['good']} mẫu")
print(f"- Trung bình (0.4-0.6): {qwen_eval['performance_distribution']['fair']} mẫu")
print(f"- Kém (<0.4): {qwen_eval['performance_distribution']['poor']} mẫu")


print("\n2. eta-llama/Llama-3.2-3B-Instruct:")
print(f"Độ chính xác: {llama_eval['accuracy']:.2%}")
print(f"Độ chênh lệch độ dài trung bình: {llama_eval['avg_length_diff']:.2f} từ")
print(f"Độ trùng lặp từ trung bình: {llama_eval['avg_word_overlap']:.2%}")
print(f"BLEU trung bình: {llama_eval['avg_bleu']:.4f}")
print(f"Độ dài câu trả lời trung bình: {llama_eval['avg_response_length']:.2f} từ")
print(f"Đa dạng từ vựng trung bình: {llama_eval['avg_vocab_diversity']:.4f}")
print(f"Tỷ lệ lặp lại từ trung bình: {llama_eval['avg_repetition_rate']:.4f}")
print("\nPhân bố chất lượng:")
print(f"- Xuất sắc (≥0.8): {llama_eval['performance_distribution']['excellent']} mẫu")
print(f"- Tốt (0.6-0.8): {llama_eval['performance_distribution']['good']} mẫu")
print(f"- Trung bình (0.4-0.6): {llama_eval['performance_distribution']['fair']} mẫu")
print(f"- Kém (<0.4): {llama_eval['performance_distribution']['poor']} mẫu")


print("\nChi tiết so sánh từng mẫu:")
for i in range(len(qwen_data)):
   print(f"\nMẫu {i+1}:")
   print(f"Câu hỏi: {qwen_data[i]['message']}")
   print(f"Câu trả lời chuẩn: {qwen_data[i]['solid_response']}")
   print("\nQwen:")
   print(f"Câu trả lời: {qwen_eval['results'][i]['chat_response']}")
   print(f"Điểm ngữ nghĩa: {qwen_eval['results'][i]['semantic_score']}")
   print(f"Độ chênh lệch độ dài: {qwen_eval['results'][i]['length_diff']}")
   print(f"Độ trùng lặp từ: {qwen_eval['results'][i]['word_overlap']}")
   print(f"BLEU: {qwen_eval['results'][i]['bleu']}")
   print(f"Độ dài câu trả lời: {qwen_eval['results'][i]['response_length']}")
   print(f"Đa dạng từ vựng: {qwen_eval['results'][i]['vocab_diversity']}")
   print(f"Tỷ lệ lặp lại từ: {qwen_eval['results'][i]['repetition_rate']}")
   print("\nLlama:")
   print(f"Câu trả lời: {llama_eval['results'][i]['chat_response']}")
