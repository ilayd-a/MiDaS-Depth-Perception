from torch.utils.data import DataLoader

# takes a dataset, batch size and shuffle variables and returns a generator
def get_dataloader(dataset, batchSize, shuffle=True):
    return DataLoader(dataset, batch_size = batchSize, shuffle = shuffle)