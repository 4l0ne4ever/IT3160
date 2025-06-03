import json
import matplotlib.pyplot as plt

def plot_scores(filename, title):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    solid_points = [r["solid_point"] for r in data["results"]]
    chatbot_points = [r["point_chatbot"] for r in data["results"]]
    indices = list(range(1, len(solid_points) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(indices, solid_points, marker='o', label='Điểm chuẩn')
    plt.plot(indices, chatbot_points, marker='x', label='Điểm chatbot')
    plt.xlabel('Cặp câu trả lời')
    plt.ylabel('Điểm số')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_scores("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/evaluation_results.json", "So sánh điểm Llama (chuẩn vs chatbot)")
plot_scores("/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/evaluation_qwen_results.json", "So sánh điểm Qwen (chuẩn vs chatbot)")