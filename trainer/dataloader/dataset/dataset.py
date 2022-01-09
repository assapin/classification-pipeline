import os


class BasicDataset:
    def __init__(self, dataset_name, dataset_id, base_dir, train_dir, val_dir, data_shape, track=False):
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.base_dir = base_dir
        self.data_shape = data_shape
        if track:
            from clearml import Dataset
            ds = Dataset.get(dataset_id)
            self.base_dir = ds.get_local_copy()
        self.train_dir = os.path.join(self.base_dir, train_dir)
        self.val_dir = os.path.join(self.base_dir, val_dir)
