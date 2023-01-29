import global_constants as gc
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


def prepare_dataloader(inputs, labels):
    data = TensorDataset(inputs, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=gc.BATCH_SIZE)

    return dataloader
