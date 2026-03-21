from base import BaseModel

class MeanBaseline(BaseModel):
    def fit(self, train_data):
        self.global_mean = train_data['rating'].mean()
        self.user_means = train_data.groupby('user')['rating'].mean()
        self.item_means = train_data.groupby('item')['rating'].mean()
        self.is_fitted_ = True

    def predict(self, user, item):
        user_mean = self.user_means.get(user, self.global_mean)
        item_mean = self.item_means.get(item, self.global_mean)
        return (user_mean + item_mean) / 2

    def predict_df(self, df):
        df['prediction'] = df.apply(lambda row: self.predict(row['user'], row['item']), axis=1)
        return df