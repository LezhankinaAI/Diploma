from libs_import import *


data_name = input()
data = pd.read_Csv(data_name)


def build_service():
    """
    Create connection with YouTube API
    """
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,
                 YOUTUBE_API_VERSION,
                 developerKey=key)


def remove_brackets(x):
    """
    Simplify response string
    """
    nstring = str(x)
    beginning_bracket = re.sub(r"'items': \[{", "'items' : {", nstring)
    ending_bracket = re.sub(r"}], 'pageInfo'", "}, 'pageInfo'", beginning_bracket)
    response_d = ast.literal_eval(ending_bracket)
    return response_d


def format_duration(duration):
    """
    Format the duration of the video into seconds
    """
    sec_patrn = re.compile(r'(\d+)S')
    min_patrn = re.compile(r'(\d+)M')
    hr_patrn = re.compile(r'(\d+)H')

    seconds = sec_patrn.search(duration)
    minutes = min_patrn.search(duration)
    hours = hr_patrn.search(duration)

    seconds = int(seconds.group(1)) if seconds else 0
    minutes = int(minutes.group(1)) if minutes else 0
    hours = int(hours.group(1)) if hours else 0

    vid_seconds = timedelta(
        hours=hours,
        minutes=minutes,
        seconds=seconds
    ).total_seconds()

    return vid_seconds


youtube = build_service()

for chunk in tqdm(range(data.shape[0] // 50 + 1)):
    if chunk * 50 + 50 < data.shape[0]:
        ids = data.video_id.values.tolist()[chunk * 50:(chunk + 1) * 50]
    else:
        ids = data.video_id.values.tolist()[chunk * 50:]
    if len(ids) > 0:
        request = youtube.videos().list(
            part="id, contentDetails, snippet",
            id=ids
        )
        response = request.execute()

        for j, resp in enumerate(response['items']):
            response_d = remove_brackets(resp)
            if 'thumbnails' in response_d['snippet']:
                data.loc[chunk * 50 + j, 'thumbnail_link'] = str(response_d['snippet']['thumbnails'])
            if 'defaultLanguage' in response_d['snippet']:
                data.loc[chunk * 50 + j, 'defaultLanguage'] = response_d['snippet']['defaultLanguage']
            if 'duration' in response_d['contentDetails']:
                data.loc[chunk * 50 + j, 'duration'] = format_duration(response_d['contentDetails']['duration'])
            if 'dimension' in response_d['contentDetails']:
                data.loc[chunk * 50 + j, 'dimension'] = response_d['contentDetails']['dimension']
            if 'definition' in response_d['contentDetails']:
                data.loc[chunk * 50 + j, 'definition'] = response_d['contentDetails']['definition']
            if 'caption' in response_d['contentDetails']:
                data.loc[chunk * 50 + j, 'caption'] = response_d['contentDetails']['caption']
            if 'regionRestriction' in response_d['contentDetails']:
                data.loc[chunk * 50 + j, 'regionRestriction'] = response_d['contentDetails']['regionRestriction']

data['Log1pViews'] = np.log1p(data['viewCount']).astype('float32')

data.loc[(pd.to_datetime(data.publishedAt).dt.hour >= 0) &
     (pd.to_datetime(data.publishedAt).dt.hour < 6), 'published_category'] = 'Night'
data.loc[(pd.to_datetime(data.publishedAt).dt.hour >= 6) &
     (pd.to_datetime(data.publishedAt).dt.hour < 12), 'published_category'] = 'Morning'
data.loc[(pd.to_datetime(data.publishedAt).dt.hour >= 12) &
     (pd.to_datetime(data.publishedAt).dt.hour < 18), 'published_category'] = 'Day'
data.loc[(pd.to_datetime(data.publishedAt).dt.hour >= 18) &
     (pd.to_datetime(data.publishedAt).dt.hour < 24), 'published_category'] = 'Evening'

data['published_weekday'] = pd.to_datetime(data.publishedAt).dt.weekday
data.published_weekday.replace({0: 'Mon', 1: 'Tue', 2: 'Wen', 3: 'Thi', 4: 'Fri', 5: 'Sat', 6: 'Sun'}, inplace=True)

data['duration_category'] = pd.qcut(data['duration'], q=5,
                                    labels=['VerySmall', 'Small', 'Medium', 'Long', 'VeryLong'])

data.loc[data['title'].str.upper() == data['title'], 'caps'] = 'Used'
data.loc[data['title'].str.upper() != data['title'], 'caps'] = 'Not used'

data.loc[(data['description'].str.find('http') != -1.0) &
         ~(data['description'].str.find('http').isna()), 'link'] = 'Used'
data.loc[(data['description'].str.find('http') == -1.0) |
         (data['description'].str.find('http').isna()), 'link'] = 'Not used'

if data_name in ['Data3_ru.csv', 'Data3_en.csv']:

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

    for i in range(1, 11):
        day = df[df.video_id.isin(data.video_id)].sort_values(['video_id', 'view_count']).groupby('video_id',
                                                                                                  as_index=False).nth(
            i - 1)
        day_target = day[['video_id', 'view_count']].reset_index(drop=True)

        data.loc[data['video_id'].isin(day_target['video_id']), f'Day{i}'] = day_target['view_count']

data.to_csv(data_name, index=False)
