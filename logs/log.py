import re
import matplotlib.pyplot as plt

with open("./logs/chatglm3/mindformer.log", "r", encoding="utf-8") as log_file:
    log_content = log_file.read()
LOSS_PATTERN = r"loss: (\d+\.\d+)"
loss_values = re.findall(LOSS_PATTERN, log_content)
loss_values = [float(loss) for loss in loss_values]
step_values = list(range(1, len(loss_values) * 2 + 1, 2))
plt.figure(figsize=(20, 6))
plt.plot(step_values, loss_values, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss vs. Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
