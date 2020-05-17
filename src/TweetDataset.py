from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        tweet = torch.from_numpy(self.X[index][0].astype(np.int32))
        label = self.y[index]
        length = self.X[index][1]

        return tweet, label, length
