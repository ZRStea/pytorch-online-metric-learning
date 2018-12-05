import numpy as np

from torch.utils.data.sampler import Sampler    

class ShopSampler(Sampler):
    def __init__(self, labels, batch_size, maxiter):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.maxiter = maxiter
        
    def __iter__(self):
        for i in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)
                subsize = len(subsample)
                if subsize <= 1:
                    continue
                subsample_size = np.random.choice(range(subsize//2, subsize+1))
                subsample = subsample[:subsample_size]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(inds)

    def __len__(self):
        return self.maxiter