from libs_import import *


categories_stats = pd.read_csv('Categories_stats.csv', encoding='utf-8')
top_stats = pd.read_csv('Top_stats.csv', encoding='utf-8')
tops = pd.read_csv('Top.csv', encoding='utf-8')
categories = pd.read_csv('Categories.csv', encoding='utf-8')

c_inds = []
for i, column in enumerate(top_stats.columns[::-1]):
    if column[1:4] == 'iew':
        c_inds.append(i)

for j in range(len(c_inds) - 1):
    top_stats[top_stats.columns[::-1][c_inds[j]]] -= top_stats[top_stats.columns[::-1][c_inds[j + 1]]]

top_stats.drop_duplicates('id', keep='first', ignore_index=True, inplace=True)
top_stats.fillna(0, inplace=True)

neg_top = set()
for row in top_stats.iterrows():
    for c, v in row[1].items():
        if c[1:4] == 'iew' and v < 0:
            neg_top.add(row[0])
            top_stats.drop(row[0], inplace=True)

c_inds = []
for i, column in enumerate(categories_stats.columns[::-1]):
    if column[1:4] == 'iew':
        c_inds.append(i)

for j in range(len(c_inds) - 1):
    categories_stats[categories_stats.columns[::-1][c_inds[j]]] -= \
        categories_stats[categories_stats.columns[::-1][c_inds[j + 1]]]

categories_stats.drop_duplicates('id', keep='first', ignore_index=True, inplace=True)

neg_cat = set()
for row in categories_stats.iterrows():
    for c, v in row[1].items():
        if c[1:4] == 'iew' and v < 0:
            neg_cat.add(row[0])
            categories_stats.drop(row[0], inplace=True)

inds_view = []
inds_like = []
inds_count = []
for i, c in enumerate(top_stats.columns[::-1]):
    if c[1:4] == 'iew':
        inds_view.append(i)
        inds_like.append(i - 1)
        inds_count.append(i - 3)

for j in range(len(inds_like) - 1):
    top_stats[top_stats.columns[::-1][inds_like[j]]] -= top_stats[top_stats.columns[::-1][inds_like[j + 1]]]
    categories_stats[categories_stats.columns[::-1][inds_like[j]]] -= categories_stats[categories_stats.columns[::-1][inds_like[j + 1]]]


for j in range(len(inds_count) - 1):
    top_stats[top_stats.columns[::-1][inds_count[j]]] -= top_stats[top_stats.columns[::-1][inds_count[j + 1]]]
    categories_stats[categories_stats.columns[::-1][inds_count[j]]] -= categories_stats[categories_stats.columns[::-1][inds_count[j + 1]]]

top = tops.loc[top_stats.index]
cat = categories.loc[categories_stats.index]

df_stats = pd.concat([top_stats, cat_stats], ignore_index=True, axis=0).drop_duplicates('id')
df_info = pd.concat([top, cat], ignore_index=True, axis=0).drop_duplicates('id')
df = pd.concat([df_stats, df_info], axis=1)
df.fillna(0, inplace=True)

df_stats.to_csv('Df_stat.csv', index=False)
df_info.to_csv('Df_info.csv', index=False)
