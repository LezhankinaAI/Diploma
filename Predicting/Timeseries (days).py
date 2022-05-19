from libs_import_dynamic import *


if not os.path.exists('active-dataset.p'):
    print('>>> Converting ACTIVE dataset from JSON format to pickle... might take a while!')
    active_videos = {}
    with bz2.BZ2File('active-dataset.json.bz2') as f:
        dataset = json.loads(f.readline())
    pickle.dump(dataset, open('active-dataset.p', 'wb'))

print('>>> Loading the ACTIVE dataset from pickle...')
active_videos = pickle.load(open('active-dataset.p', 'rb'))
df = pd.DataFrame(active_videos)

df = df[['YoutubeID', 'watchTime', 'dailyViewcount', 'description',
        'title', 'channelId', 'channelTitle', 'category', 'uploadDate',
        'duration', 'definition', 'dimension', 'caption', 'regionRestriction.blocked',
        'regionRestriction.allowed', 'topicIds', 'relevantTopicIds']]

min_days = min(df['watchTime'].apply(lambda x: len(x)))
days = [f'day_{i}' for i in range(1, min_days + 1)]
timeline = pd.DataFrame(index=df['YoutubeID'], columns=days)

for i in tqdm(range(min_days)):
    timeline[f'day_{i + 1}'] = df['dailyViewcount'].str[i].values.astype('int64')

pred_val = int(input())
drops = [f'day_{i}' for i in range(pred_val, 120)]
y = timeline[drops[0]]
X = timeline.drop(columns=drops)

X['Month'] = pd.to_datetime(df[df.YoutubeID.isin(X.index)].uploadDate).dt.month.values
X['Day'] = pd.to_datetime(df[df.YoutubeID.isin(X.index)].uploadDate).dt.day.values
X['Hour'] = pd.to_datetime(df[df.YoutubeID.isin(X.index)].uploadDate).dt.hour.values
X['Weekday'] = pd.to_datetime(df[df.YoutubeID.isin(X.index)].uploadDate).dt.weekday.values

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

X['duration'] = df[df.YoutubeID.isin(X.index)]['duration'].apply(format_duration).values

X['title'] = df[df.YoutubeID.isin(X.index)]['title'].values
X['description'] = df[df.YoutubeID.isin(X.index)]['description'].values
X['definition'] = df[df.YoutubeID.isin(X.index)]['definition'].values
X['definition'].fillna('Not stated', inplace=True)
X['category'] = df[df.YoutubeID.isin(X.index)]['category'].values
X['category'].fillna('Not stated', inplace=True)

X.loc[X['title'].str.upper() == X['title'], 'caps'] = 'Used'
X.loc[X['title'].str.upper() != X['title'], 'caps'] = 'Not used'

X.loc[(X['description'].str.find('http') != -1.0) &
        ~(X['description'].str.find('http').isna()), 'link'] = 'Used'
X.loc[(X['description'].str.find('http') == -1.0) |
        (X['description'].str.find('http').isna()), 'link'] = 'Not used'

X['min'] = X[X.columns[:pred_val - 1]].min(axis=1)
X['max'] = X[X.columns[:pred_val - 1]].max(axis=1)
X['mean'] = X[X.columns[:pred_val - 1]].mean(axis=1)


def approx(x : np.array, a : float, b : float, c : float) -> np.array:
    """
    Counts approximate trend line
    """
    return a / (np.exp(b * x) + c)


ids = []
parameters = {}
for row in tqdm(X.iterrows()):
    try:
        value = X.loc[row[0]][X.columns[:pred_val - 1]].values
        x_line = np.arange(0, pred_val - 1)
        params, _ = curve_fit(approx, x_line, value, maxfev=100000)
        y_line = approx(x_line, *params)
        parameters[row[0]] = params
        X.loc[row[0], X.columns[:pred_val - 1]] = value - y_line
    except:
        ids.append(row[0])
        continue

X = X[~X.index.isin(ids)].fillna(0)
y = y[~y.index.isin(ids)].fillna(0).astype('float64')

for i in range(len(y)):
    y[i] -= approx(pred_val, *parameters[y.index[i]])

categorical = ['category', 'Month', 'Weekday', 'Hour', 'definition', 'link', 'caps']

numeric_features = ['min', 'max', 'mean', 'duration', *X.columns[:pred_val - 1].to_list()]

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric_features)
])


def testing(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    model.fit(X_train[[*numeric_features, *categorical]], y_train)
    test = y_test.copy()
    y_pred = model.predict(X_test[[*numeric_features, *categorical]])
    for i in range(len(y_test)):
        val = approx(pred_val, *parameters[X_test.index[i]])
        test[i] += val
        y_pred[i] += val
    return y_pred, test


batch = floor(X.shape[0] / 4)

data = pd.DataFrame(index=X.index, columns=['Prediction'])

for i in range(1, 5):
    X_test = X[batch * (i - 1): batch * i]
    y_test = y[X_test.index]
    X_valid = X.loc[np.random.choice(X.drop(index=X_test.index.tolist()).index,
                                     batch, replace=False)]
    y_valid = y[X_valid.index]
    X_train = X.drop(index=[*X_test.index.to_list(), *X_valid.index.to_list()])
    y_train = y[X_train.index]

    model = Pipeline(steps=[
                            ('ohe_and_scaling', column_transformer),
                            ('regression', TheilSenRegressor(random_state=21))])

    curr_pred, curr_test = testing(model, X_train, y_train, X_test, y_test)
    data.loc[X_test.index, 'Prediction'] = curr_pred

data.to_csv('2_1_Prediction.csv')
