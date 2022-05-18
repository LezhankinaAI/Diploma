from libs_import import *
from functions import *

tops = pd.read_csv('Top.csv', encoding='utf-8')
categories = pd.read_csv('Categories.csv', encoding='utf-8')

tops_stats = tops[['id', 'viewCount', 'likeCount', 'favoriteCount', 'commentCount']]
tops_stats.columns = ['id', f'viewCount_{yesterday}', f'likeCount_{yesterday}',
               f'favoriteCount_{yesterday}', f'commentCount_{yesterday}']
tops_stats.to_csv('Top_stats.csv', encoding='utf-8', index=False)

categories_stats = categories[['id', 'viewCount', 'likeCount', 'favoriteCount', 'commentCount']]
categories_stats.columns = ['id', f'viewCount_{yesterday}', f'likeCount_{yesterday}',
               f'favoriteCount_{yesterday}', f'commentCount_{yesterday}']
categories_stats.to_csv('Categories_stats.csv', encoding='utf-8', index=False)

file_names = ['Top', 'Categories']

youtube = build_service()
today = date.today()

for i, curr_df in enumerate([tops, categories]):
    csv_input = pd.read_csv(f'{file_names[i]}_stats.csv', encoding='utf-8')
    for chunk in tqdm(range(curr_df.shape[0] // 50 + 1)):
        if chunk * 50 + 50 < curr_df.shape[0]:
            ids = curr_df.id.values.tolist()[chunk * 50:(chunk + 1) * 50]
        else:
            ids = curr_df.id.values.tolist()[chunk * 50:]
        if len(ids) > 0:
            request = youtube.videos().list(
                part="id, statistics",
                id=ids
            )
            response = request.execute()

            for j, resp in enumerate(response['items']):
                response_d = remove_brackets(resp)
                if 'viewCount' in response_d['statistics']:
                    csv_input.loc[chunk * 50 + j, f'viewCount_{today}'] = response_d['statistics']['viewCount']
                if 'likeCount' in response_d['statistics']:
                    csv_input.loc[chunk * 50 + j, f'likeCount_{today}'] = response_d['statistics']['likeCount']
                if 'favoriteCount' in response_d['statistics']:
                    csv_input.loc[chunk * 50 + j, f'favoriteCount{today}'] = response_d['statistics']['favoriteCount']
                if 'commentCount' in response_d['statistics']:
                    csv_input.loc[chunk * 50 + j, f'commentCount{today}'] = response_d['statistics']['commentCount']

    csv_input.to_csv(f'{file_names[i]}_stats.csv', encoding='utf-8', index=False)
