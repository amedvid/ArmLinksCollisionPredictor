import time

import torch
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collision_threshold = 0.0

def test_model(model: nn.Module, test_loader: DataLoader, criterion):
    model.eval()

    real_false_ans_false = 0
    real_false_ans_true = 0
    real_true_ans_false = 0
    real_true_ans_true = 0

    fully_correct_count = 0
    partially_wrong_count = 0
    start_time = time.time()

    with torch.no_grad():
        test_loss = 0.0
        total_samples = 0
        for features, labels in test_loader:
            features = features.to(device)   # (B, D)
            labels = labels.to(device)       # (B, 8)

            outputs = model(features)        # (B, 8)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            predictions = outputs > collision_threshold
            actual = labels > collision_threshold

            all_correct = (predictions == actual).all(dim=-1)
            any_wrong = ~all_correct

            fully_correct_count += all_correct.sum().item()
            partially_wrong_count += any_wrong.sum().item()

            real_false_ans_false += ((labels < collision_threshold) & (outputs < collision_threshold)).sum().item()
            real_false_ans_true  += ((labels < collision_threshold) & (outputs > collision_threshold)).sum().item()
            real_true_ans_false  += ((labels > collision_threshold) & (outputs < collision_threshold)).sum().item()
            real_true_ans_true   += ((labels > collision_threshold) & (outputs > collision_threshold)).sum().item()

        avg_loss = test_loss / total_samples if total_samples > 0 else 0.0
        total = real_false_ans_false + real_false_ans_true + real_true_ans_false + real_true_ans_true
        total_success = (real_true_ans_true + real_false_ans_false) / total if total > 0 else 0.0
        correct_answers_count = float(fully_correct_count) / (fully_correct_count + partially_wrong_count) if (fully_correct_count + partially_wrong_count) > 0 else 0.0

        print(f"Test Loss: {avg_loss:.8f}\n"
              f"TT:{real_true_ans_true}  FF:{real_false_ans_false}\n"
              f"TF:{real_true_ans_false}  FT:{real_false_ans_true}\n"
              f"total success: {total_success * 100:.3f}\n"
              f"+:{fully_correct_count}  -:{partially_wrong_count}\n"
              f"total success: {correct_answers_count * 100:.3f}")

    end_time = time.time()
    execution_duration = end_time - start_time
    print(f"execution time: {execution_duration:.4f}s\n")
