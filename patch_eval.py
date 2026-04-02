import re

with open('utils/eval_metrics_train.py', 'r') as f:
    content = f.read()

# 1. Add compute_backward_transfer to defaults
content = content.replace('"compute_accuracy": True,', '"compute_accuracy": True,\n            "compute_backward_transfer": True,')
content = content.replace('"compute_accuracy": False,', '"compute_accuracy": False,\n        "compute_backward_transfer": False,')

# 2. Add self.past_sample_accuracies in Evaluator.__init__
init_code = """
        # Sliding window cursor — tracks position within the eval pool
        self._eval_cursor = 0
        self.past_sample_accuracies = {}  # Tracks max accuracy per eval sample for BWT
"""
content = re.sub(r'# Sliding window cursor.*?self\._eval_cursor = 0', init_code.strip(), content, flags=re.DOTALL)

# 3. Add backward transfer calculation in evaluate
accuracy_code = """
                if mf.get("compute_accuracy", False):
                    correct_list = [g == p for g, p in zip(gold_labels, pred_labels)]
                    correct = sum(correct_list)
                    metrics["accuracy"] = correct / total
                    _log(f"  Correct: {correct} / {total}")

                    if mf.get("compute_backward_transfer", False):
                        bwt_sum = 0.0
                        bwt_count = 0
                        for i, is_correct in enumerate(correct_list):
                            sample_idx = (window_start + i) % len(self.eval_data)
                            curr_acc = 1.0 if is_correct else 0.0
                            
                            if sample_idx in self.past_sample_accuracies:
                                max_past = self.past_sample_accuracies[sample_idx]
                                bwt_sum += (curr_acc - max_past)
                                bwt_count += 1
                                if curr_acc > max_past:
                                    self.past_sample_accuracies[sample_idx] = curr_acc
                            else:
                                self.past_sample_accuracies[sample_idx] = curr_acc
                                
                        if bwt_count > 0:
                            metrics["backward_transfer"] = bwt_sum / bwt_count
"""
content = re.sub(r'if mf\.get\("compute_accuracy", False\):.*?_log\(f"  Correct: \{correct\} / \{total\}"\)', accuracy_code.strip(), content, flags=re.DOTALL)

with open('utils/eval_metrics_train.py', 'w') as f:
    f.write(content)
