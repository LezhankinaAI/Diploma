from libs_import import *
from functions import *


with open('country-codes.txt') as f:
    regions = f.read().splitlines()

youtube = build_service()
video_data_entries = []
for region in tqdm(regions):
    request = youtube.videos().list(
        part="id, contentDetails, snippet, statistics",
        chart="mostPopular",
        regionCode=region,
        maxResults=50,
        pageToken=None)
    response = request.execute()

    request2 = youtube.videoCategories().list(
        part="snippet",
        regionCode=region
    )
    response2 = request2.execute()

    list_of_category_info = response2['items']

    for resp in response['items']:
        response_d = remove_brackets(resp)

        video_data_entries = collect_data(response_d)

columns = ['id', 'viewCount', 'likeCount', 'favoriteCount', 'commentCount', 'duration', 'dimension',
           'definition', 'caption', 'licensedContent', 'contentRating', 'projection', 'publishedAt',
           'channelId', 'title', 'description', 'thumbnails', 'channelTitle', 'tags', 'categoryId',
           'liveBroadcastContent', 'defaultAudioLanguage', 'defaultLanguage', 'regionRestriction', 'date']

export(video_data_entries, 'Top.csv', columns)

with open('category-codes.txt') as f:
    categories = f.read().splitlines()

youtube = build_service()
video_data_entries = []
for region in tqdm(regions):
    for category in categories:
        try:
            request = youtube.videos().list(
                part="id, contentDetails, snippet, statistics",
                chart='mostPopular',
                regionCode=region,
                videoCategoryId=category.split(' - ')[0],
                maxResults=10
            )
            response = request.execute()
        except HttpError:
            continue

        request2 = youtube.videoCategories().list(
            part="snippet",
            regionCode=region
        )
        response2 = request2.execute()

        list_of_category_info = response2['items']

        for resp in response['items']:
            response_d = remove_brackets(resp)

            video_data_entries = collect_data(response_d)

export(video_data_entries, 'Categories.csv', columns)
