import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 불러오기
file_path = './box plot.xlsx'
df = pd.read_excel(file_path)

# 컬럼 이름 정리
df.columns = ['Problem', 'Lin', 'Ours']

# 스케일명 추출 함수 정의
def extract_scale(problem_name):
    return problem_name.split('_')[0] + '_' + problem_name.split('_')[1]

df['Scale'] = df['Problem'].apply(extract_scale)

# 스케일별 box plot을 1행 8열 subplot으로 출력
scales = df['Scale'].unique()

fig, axes = plt.subplots(1, 8, figsize=(10, 2.5), sharey=True)
palette = {'Lin': 'tab:blue', 'Ours': 'tab:orange'}

for i, scale in enumerate(scales):
    sub_df = df[df['Scale'] == scale]
    melted = sub_df.melt(id_vars=['Problem'], value_vars=['Lin', 'Ours'], var_name='Method', value_name='Gap')
    sns.boxplot(x='Method', y='Gap', data=melted, ax=axes[i], palette=palette)
    axes[i].set_title(scale, fontsize=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('' if i > 0 else 'Gap (%)')

plt.tight_layout()
plt.savefig("./box.pdf", format="pdf", bbox_inches="tight")
plt.show()
