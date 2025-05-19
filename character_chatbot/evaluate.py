import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Kiểm tra MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Tải dữ liệu từ file JSON
with open("chatbot_results.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Tải mô hình reranker
model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Giảm kích thước mô hình
    device_map="auto"  # Tự động dùng MPS
)
model.eval()
model.to(device)

# Hàm chấm điểm
def score_pairs(pairs, tokenizer, model, device):
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512  # Giới hạn độ dài đầu vào
        ).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    return scores.cpu().numpy()

# Đánh giá độ chính xác
correct = 0
threshold = 0.1
results = []
for sample in test_data:
    message = sample["message"]
    chat_response = sample["chat_response"]
    solid_response = sample["solid_response"]
    
    pairs = [
        [message, chat_response],
        [message, solid_response]
    ]
    scores = score_pairs(pairs, tokenizer, model, device)
    
    is_correct = abs(scores[0] - scores[1]) <= threshold or scores[0] >= scores[1]
    correct += 1 if is_correct else 0
    
    results.append({
        "message": message,
        "chat_response": chat_response,
        "solid_response": solid_response,
        "point_chatbot": float(scores[0]),
        "solid_point": float(scores[1]),
        "accuracy": is_correct
    })
    
    # Giải phóng bộ nhớ
    torch.mps.empty_cache() if device.type == "mps" else None

accuracy = correct / len(test_data)
print(f"Chatbot accuracy: {accuracy:.2%}")

# Lưu kết quả đánh giá
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)

# Giải phóng tài nguyên
del model
del tokenizer
torch.mps.empty_cache() if device.type == "mps" else None

# Evaluate the model
with open("evaluation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
avg_diff = sum(r["solid_point"] - r["point_chatbot"] for r in data["results"]) / len(data["results"])
print(f"Chênh lệch điểm trung bình (chuẩn - chatbot): {avg_diff:.4f}")