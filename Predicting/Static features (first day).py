from libs_import_static import *


data_name = input()
data = pd.read_csv(data_name)

replacer = dict()

with open('category-codes.txt') as f:
    categories = f.read().splitlines()
    for cat in categories:
        replacer[int(cat.split(' - ')[0])] = cat.split(' - ')[1]

text_columns = ['title', 'description', 'tags']
categorical_columns = ['categoryId', 'published_weekday', 'published_category', 'caps', 'link', 'duration_category']
TARGET_COLUMN = "Log1pViews"

data['categoryId'] = data['categoryId'].replace(replacer)
data['duration_category'] = data['duration_category'].fillna('Medium')
data[categorical_columns[:-1]] = data[categorical_columns[:-1]].fillna('NaN')
data['title'] = data['title'].fillna('NaN')
data['description'] = data['description'].str.replace(r'http\S+', '').fillna('NaN')
data['tags'] = data['tags'].str.replace('|', ', ').replace('[None]', 'NaN')

tokenizer = nltk.tokenize.WordPunctTokenizer()
data["description"] = data["description"].apply(lambda x: ' '.join(tokenizer.tokenize(x.lower())))
data["title"] = data["title"].apply(lambda x: ' '.join(tokenizer.tokenize(str(x).lower())))
data["tags"] = data["tags"].apply(lambda x: ' '.join(''.join(str(x).lower().split(' ')).split(',')))

token_counts = Counter()

for row in data.iterrows():
    token_counts.update([*row[1]['description'].split(), *row[1]['title'].split(), *row[1]['tags'].split()])

min_count = 5
tokens = sorted(t for t, c in token_counts.items() if c >= min_count)
UNK, PAD = "UNK", "PAD"
tokens = [UNK, PAD] + tokens

token_to_id = dict(zip(tokens, [i for i in range(len(tokens))]))

UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])


def as_matrix(sequences, max_len=None):
    """
    Convert a list of tokens into a matrix with padding
    """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))

    max_len = min(max(map(len, sequences)), max_len or float('inf'))

    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i, seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix

categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
categorical_vectorizer.fit(data[categorical_columns].apply(dict, axis=1))

data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
data_train.index = range(len(data_train))
data_val.index = range(len(data_val))

DEVICE = torch.device('cuda')


def to_tensors(batch, device):
    batch_tensors = dict()
    for key, arr in batch.items():
        if key in ["description", "title", 'tags']:
            batch_tensors[key] = torch.tensor(arr, dtype=torch.int64, device=device)
        else:
            batch_tensors[key] = torch.tensor(arr, device=device)
    return batch_tensors


def make_batch(data, max_len=None, word_dropout=0, device=DEVICE):
    batch = {}
    batch["title"] = as_matrix(data["title"].values, max_len)
    batch["description"] = as_matrix(data["description"].values, max_len)
    batch["tags"] = as_matrix(data["tags"].values, max_len)
    batch['Categorical'] = categorical_vectorizer.transform(data[categorical_columns].apply(dict, axis=1))

    if word_dropout != 0:
        batch["description"] = apply_word_dropout(batch["description"], 1. - word_dropout)

    if TARGET_COLUMN in data.columns:
        batch[TARGET_COLUMN] = data[TARGET_COLUMN].values

    return to_tensors(batch, device)


def apply_word_dropout(matrix, keep_prop, replace_with=UNK_IX, pad_ix=PAD_IX, ):
    dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop])
    dropout_mask &= matrix != pad_ix
    return np.choose(dropout_mask, [matrix, np.full_like(matrix, replace_with)])


class ViewsPredictor(nn.Module):
    def __init__(self, n_tokens=len(tokens), n_cat_features=len(categorical_vectorizer.vocabulary_), hid_size=64):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=n_tokens,
                                      embedding_dim=hid_size,
                                      padding_idx=0,
                                      max_norm=5.0).to(DEVICE)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=100,
                                              kernel_size=(fs, hid_size)) for fs in [1, 2, 3, 4]]).to(DEVICE)

        self.lstm = nn.LSTM(64, hidden_size=hid_size, batch_first=True, bidirectional=True).to(DEVICE)
        self.fc = nn.Linear(hid_size * 2, 100).to(DEVICE)

        self.hidden = nn.Linear(200, 100).to(DEVICE)

        self.max_pool = nn.AdaptiveMaxPool1d(2).to(DEVICE)
        self.avg_pool = nn.AdaptiveAvgPool1d(2).to(DEVICE)

        self.categories1 = nn.Linear(n_cat_features, 300).to(DEVICE)
        self.norm_cat = nn.BatchNorm1d(300, momentum=0.1).to(DEVICE)
        self.categories2 = nn.Linear(300, 100).to(DEVICE)

        self.relu = nn.ReLU().to(DEVICE)
        self.dropout = nn.Dropout(0.5).to(DEVICE)

        self.fc1 = nn.Linear(100 * 49, hid_size, bias=True).to(DEVICE)
        self.fc2 = nn.Linear(hid_size, 1, bias=True).to(DEVICE)

    def forward(self, batch):
        # emb
        x_embed_title = self.embedding(batch['title']).unsqueeze(1)
        x_embed_description = self.embedding(batch['description']).unsqueeze(1)
        x_embed_tags = self.embedding(batch['tags']).unsqueeze(1)

        # conv
        convolution_title = [conv(x_embed_title) for conv in self.convs]
        convolution_description = [conv(x_embed_description) for conv in self.convs]
        convolution_tags = [conv(x_embed_tags) for conv in self.convs]

        # pooling
        max_title = torch.cat([self.max_pool(conv_title.squeeze().unsqueeze(2 if batch['title'].shape[0] != 1 else 0))
                               if len(conv_title.squeeze().shape) < 3 else self.max_pool(conv_title.squeeze())
                               for conv_title in convolution_title], dim=2)

        avg_title = torch.cat([self.avg_pool(conv_title.squeeze().unsqueeze(2 if batch['title'].shape[0] != 1 else 0))
                               if len(conv_title.squeeze().shape) < 3 else self.avg_pool(conv_title.squeeze())
                               for conv_title in convolution_title], dim=2)

        title = torch.cat([max_title, avg_title], dim=-1)

        max_description = torch.cat(
            [self.max_pool(conv_description.squeeze().unsqueeze(2 if batch['description'].shape[0] != 1 else 0))
             if len(conv_description.squeeze().shape) < 3 else self.max_pool(conv_description.squeeze())
             for conv_description in convolution_description], dim=2)

        avg_description = torch.cat(
            [self.avg_pool(conv_description.squeeze().unsqueeze(2 if batch['description'].shape[0] != 1 else 0))
             if len(conv_description.squeeze().shape) < 3 else self.avg_pool(conv_description.squeeze())
             for conv_description in convolution_description], dim=2)

        description = torch.cat([max_description, avg_description], dim=-1)

        max_tags = torch.cat([self.max_pool(conv_tags.squeeze().unsqueeze(2 if batch['tags'].shape[0] != 1 else 0))
                              if len(conv_tags.squeeze().shape) < 3 else self.max_pool(conv_tags.squeeze())
                              for conv_tags in convolution_tags], dim=2)

        avg_tags = torch.cat([self.avg_pool(conv_tags.squeeze().unsqueeze(2 if batch['tags'].shape[0] != 1 else 0))
                              if len(conv_tags.squeeze().shape) < 3 else self.avg_pool(conv_tags.squeeze())
                              for conv_tags in convolution_tags], dim=2)

        tags = torch.cat([max_tags, avg_tags], dim=-1)

        # categories
        categories = self.categories2(self.norm_cat(self.categories1(batch['Categorical']))).unsqueeze(1).permute(0, 2,
                                                                                                                  1)

        # concat
        cat = torch.cat((title, description, tags, categories), dim=2)

        out = cat.view(cat.shape[0], -1)
        out = self.fc1(self.relu(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.reshape(-1)


def iterate_minibatches(data, batch_size=256, shuffle=True, cycle=False, device=DEVICE, **kwargs):
    """ iterates minibatches of data in random order """
    while True:
        indices = np.arange(len(data))
        if shuffle:
            indices = np.random.permutation(indices)

        for start in range(0, len(indices), batch_size):
            batch = make_batch(data.iloc[indices[start: start + batch_size]], **kwargs)
            yield batch

        if not cycle: break

BATCH_SIZE = 16
EPOCHS = 16

df = pd.DataFrame(index=data_val.video_id, columns=['Prediction'])


def get_pred(model, data, batch_size=BATCH_SIZE, name="", **kw):
    model.eval()
    with torch.no_grad():
        for batch in iterate_minibatches(data, batch_size=batch_size, shuffle=False, **kw):
            batch_pred = model(batch)
            df.loc[batch['video_id'], 'Prediction'] = batch_pred
    return df


model = ViewsPredictor().to(DEVICE)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    print(f"epoch: {epoch}")
    model.train()
    for i, batch in tqdm(enumerate(
            iterate_minibatches(data_train, batch_size=BATCH_SIZE, device=DEVICE)),
            total=len(data_train) // BATCH_SIZE
    ):
        pred = model(batch)
        loss = criterion(pred, batch[TARGET_COLUMN].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    df = get_pred(model, data_val)

df.to_csv(f'1_1_Prediction_{data_name[-6:-4]}.csv')
