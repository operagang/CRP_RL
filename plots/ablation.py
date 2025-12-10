import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 설정
file_path = "./plots/ablation(R).xlsx"
excel_data = pd.read_excel(file_path, sheet_name=None)
window = 10  # moving average window
font_size = 15  # 폰트 사이즈 설정
# 순서를 명시적으로 지정 (Proposed → 나머지)
# ordered_keys = sorted(excel_data.keys(), key=lambda x: 0 if x == "Proposed" else 1)
ordered_keys = ['no SD learning', 'no Attn.', 'no LSTM', 'Proposed']

# 그래프 시작
plt.figure(figsize=(8, 6))

for method_name in ordered_keys:
    df = excel_data[method_name]
    # 첫 열 제거 후 수치 데이터만
    data = df.iloc[:, 1:].to_numpy()

    # row-wise 평균, std 계산 (→ epoch 별)
    mean_values = np.nanmean(data, axis=0)
    std_values = np.nanstd(data, axis=0)

    # safer moving average using pandas
    def moving_average_safe(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    mean_ma = moving_average_safe(mean_values, window)
    std_ma = moving_average_safe(std_values, window)
    epochs = np.arange(len(mean_ma))

    label = method_name.replace("no", "w/o")
    plt.plot(epochs, mean_ma, label=label, linewidth=2)
    plt.fill_between(epochs, mean_ma - std_ma, mean_ma + std_ma, alpha=0.15)

# Plot 설정
plt.xlim(0, 100)
plt.ylim(5, 30)
plt.xlabel("Epoch", fontsize=font_size +1)
plt.ylabel("Gap (%)", fontsize=font_size +1)
plt.title("Average Gap on the Random Benchmark", fontsize=font_size + 2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size - 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("./plots/ablation(R).pdf", format="pdf", bbox_inches="tight")
