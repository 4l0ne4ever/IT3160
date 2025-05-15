from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate
import torch

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df, device='cpu'):
    # Get unique labels and convert to NumPy array
    classes = np.array(sorted(df['label_dtframe'].unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df['label_dtframe'].tolist()
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)