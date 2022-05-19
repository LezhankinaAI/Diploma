from libs_import_dynamic import *


df_stats = pd.read_csv('Df_stat.csv')
df_info = pd.read_csv('Df_info.csv')
df = pd.concat([df_stats, df_info], axis=1)
df.fillna(0, inplace=True)

X = df[[*df_stats.columns.to_list(), 'duration', 'definition', 'publishedAt', 'title', 'description', 'categoryId']]
X.index = df.iloc[:, 0]
X.loc[:, 'Month'] = pd.to_datetime(X.publishedAt).dt.month.values
X.loc[:, 'Day'] = pd.to_datetime(X.publishedAt).dt.day.values
X.loc[:, 'Hour'] = pd.to_datetime(X.publishedAt).dt.hour.values
X.loc[:, 'Weekday'] = pd.to_datetime(X.publishedAt).dt.weekday.values
X.loc[(X['title'].str.upper() == X['title']), 'caps'] = 'Used'
X.loc[(X['title'].str.upper() != X['title']), 'caps'] = 'Not used'
X.loc[(X['description'].str.find('http') != -1.0) &
      ~(X['description'].str.find('http').isna()), 'link'] = 'Used'
X.loc[(X['description'].str.find('http') == -1.0) |
      (X['description'].str.find('http').isna()), 'link'] = 'Not used'

pred_val = 12 + 14  # last is 30, first is 13
target = f'viewCount_2022-04-{pred_val}'
y = X[target]

drop_cols = []
for column in X.columns:
    if column[-2:].isdigit() and int(column[-2:]) >= pred_val:
        drop_cols.append(column)

X.drop(columns=[*drop_cols], inplace=True)
X.fillna(0, inplace=True)

categorical = ['categoryId', 'Month', 'Weekday', 'Hour', 'definition', 'link', 'caps']

numeric_features = X.columns.drop([*categorical, 'title', 'description', 'publishedAt', 'Day'])

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric_features)
])

def testing(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):

    model.fit(X_train[[*numeric_features, *categorical]], y_train)
    test = y_test.copy()
    y_pred = model.predict(X_test[[*numeric_features, *categorical]])

    return y_pred, test


batch = floor(X.shape[0] / 4)
print('Batch size:', batch)

data = pd.DataFrame(index=X.index, columns=['Prediction'])

for i in range(1, 5):
    X_test = X[batch * (i - 1): batch * i]
    y_test = y[X_test.index]
    X_valid = X.loc[np.random.choice(X.drop(index=X_test.index.tolist()).index,
                                     batch, replace=False)]
    y_valid = y[X_valid.index]
    X_train = X.drop(index=[*X_test.index.to_list(), *X_valid.index.to_list()])
    y_train = y[X_train.index]

    max_features = [int(np.sqrt(X_train.shape[1])), X_train.shape[1] // 2, X_train.shape[1] - 1]
    n_estimators = [25, 50, 100]

    searcher = GridSearchCV(Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', BaggingRegressor(random_state=21))]),
        [{"regression__n_estimators": n_estimators, "regression__max_features": max_features}],
        scoring='neg_mean_squared_error', cv=4)
    searcher.fit(X_valid[[*numeric_features, *categorical]], y_valid)
    model = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', BaggingRegressor(random_state=21,
                                        n_estimators=searcher.best_params_["regression__n_estimators"],
                                        max_features=searcher.best_params_["regression__max_features"]))])

    curr_pred, curr_test = testing(model, X_train, y_train, X_test, y_test)
    data.loc[X_test.index, 'Prediction'] = curr_pred

data.to_csv('2_2_Prediction.csv')
