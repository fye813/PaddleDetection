import pandas as pd

# CSVファイルを読み込む
file = 'output/2024-1129_1530/excluded_data_.csv'
df = pd.read_csv(file)
file = '/mnt/d/work/excluded_data_1127.csv'
df_1127 = pd.read_csv(file)


ids = [
# A
1387,
1417
# C
# 7316,
# 7369,
# 7777,
# 7824,
# 8738,
# 9037,
# 9260,
# 9415,
# 9699,
# 10093,
# 10836,
# 10873
# D
# 35127,
# 35833,
# 35916,
# 36628,
# 40661,
# 42606,
# 49809,
# 50298
# E
# 122723,
# 124605,
# 124650
]

df = df[df["original_ID"].isin(ids)]
df_1127 = df_1127[df_1127["original_ID"].isin(ids)]

# Detection IDごとにPlaceの数をカウント
place_counts = df.groupby('original_ID')['Place'].unique()
# 結果を表示
print("各Detection IDごとのPlace数:")
print(place_counts)


df.to_csv("output/excluded_data_filter.csv",index=False)
df_1127.to_csv("output/excluded_data_1127_filter.csv",index=False)
