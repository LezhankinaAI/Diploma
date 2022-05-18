from libs_import import *


# IN, US, GB, DE, CA, FR, RU, BR, MX, KR, JP

not_codes = ['KR', 'JP']
columns = ['video_id', 'title', 'publishedAt', 'channelId', 'channelTitle',
       'categoryId', 'trending_date', 'tags', 'view_count', 'likes',
       'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
       'ratings_disabled', 'description', 'region']
df = pd.DataFrame(columns=columns)

with zipfile.ZipFile('archive2.zip', 'r') as zf:
    for file in zf.namelist():
        if file[:2] in not_codes or not file.endswith('.csv'):
            continue
        print(file)
        rg = pd.read_csv(zf.open(file))
        rg['region'] = file[:2]
        df = pd.concat([df, rg], ignore_index=True)

first_day = df[pd.to_datetime(df.trending_date).dt.date == pd.to_datetime(df.publishedAt).dt.date + timedelta(1)].reset_index(drop=True)
first_day.drop_duplicates('video_id', keep='last', inplace=True)
first_day_target = first_day[['video_id', 'view_count']].reset_index(drop=True)

data1 = df[df.video_id.isin(first_day_target.video_id) & df.view_count.isin(first_day_target.view_count)]\
    .drop_duplicates('video_id', keep='last').reset_index()

detector = Translator()

data1.loc[data1['region'].isin(['CA', 'GB', 'US']), 'lang'] = 'en'
data1.loc[data1['region'].isin(['RU']), 'lang'] = 'ru'

for row in tqdm(data1[data1.lang.isna()].iterrows()):
    try:
        data1.loc[row[0], 'lang'] = str(detector.detect(row[1]['description'][:25]).lang)
    except TypeError:
        data1.loc[row[0], 'lang'] = str(detector.detect(row[1]['title']).lang)

data1_en = data1[data1['lang'] == 'en'].reset_index(drop=True)
data1_ru = data1[data1['lang'] == 'ru'].reset_index(drop=True)


seventh_day = df[pd.to_datetime(df.trending_date).dt.date == pd.to_datetime(df.publishedAt).dt.date + timedelta(7)].reset_index()
seventh_day.drop_duplicates('video_id', keep='last', inplace=True)
seventh_day_target = seventh_day[['video_id', 'view_count']].reset_index(drop=True)
seventh_day_target.loc[:, 'view_count'] /= 7

data2 = df[df.video_id.isin(seventh_day_target.video_id)].drop_duplicates('video_id', keep='last').reset_index()
data2['view_count'] = seventh_day_target.view_count

data2.loc[data2['region'].isin(['CA', 'GB', 'US']), 'lang'] = 'en'
data2.loc[data2['region'].isin(['RU']), 'lang'] = 'ru'

for row in tqdm(data2[data2.lang.isna()].iterrows()):
    try:
        data2.loc[row[0], 'lang'] = str(detector.detect(row[1]['description'][:25]).lang)
    except TypeError:
        data2.loc[row[0], 'lang'] = str(detector.detect(row[1]['title']).lang)

data2_en = data2[data2['lang'] == 'en'].reset_index(drop=True)
data2_ru = data2[data2['lang'] == 'ru'].reset_index(drop=True)


data3 = df.copy()
data3['trending_date'] = pd.to_datetime(data3['trending_date']).dt.date
vals = data3.groupby('video_id')['trending_date'].nunique().values
data3.drop_duplicates('video_id', keep='last', inplace=True)
data3.sort_values('video_id', inplace=True)
data3.reset_index(inplace=True, drop=True)
data3['view_count'] = vals

data3.loc[data3['region'].isin(['CA', 'GB', 'US']), 'lang'] = 'en'
data3.loc[data3['region'].isin(['RU']), 'lang'] = 'ru'

for row in tqdm(data3[data3.lang.isna()].iterrows()):
    try:
        data3.loc[row[0], 'lang'] = str(detector.detect(row[1]['description'][:25]).lang)
    except TypeError:
        data3.loc[row[0], 'lang'] = str(detector.detect(row[1]['title']).lang)

data3_en = data3[data3['lang'] == 'en'].reset_index(drop=True)
data3_ru = data3[data3['lang'] == 'ru'].reset_index(drop=True)

data1_en.to_csv('Data1_en.csv', index=False)
data1_ru.to_csv('Data1_ru.csv', index=False)
data2_en.to_csv('Data2_en.csv', index=False)
data2_ru.to_csv('Data2_ru.csv', index=False)
data3_en.to_csv('Data3_en.csv', index=False)
data3_ru.to_csv('Data3_ru.csv', index=False)
