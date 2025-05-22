import json

with open("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/evaluation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
avg_diff = sum(r["solid_point"] - r["point_chatbot"] for r in data["results"]) / len(data["results"])
print(f"Chênh lệch điểm trung bình Llama (chuẩn - chatbot): {avg_diff:.4f}")

with open("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/evaluation_qwen_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
avg_diff = sum(r["solid_point"] - r["point_chatbot"] for r in data["results"]) / len(data["results"])
print(f"Chênh lệch điểm trung bình Qwen (chuẩn - chatbot): {avg_diff:.4f}")