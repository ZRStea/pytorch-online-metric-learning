import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
from sklearn.cluster import KMeans,MiniBatchKMeans
class Evaluation(nn.Module):
    def __init__(self, df_test, df_query, dataloader_test, dataloader_query, cuda):
        self.test_labels = np.array(df_test['label'])
        self.query_labels = np.expand_dims(np.array(df_query['label']), 1)
        self.test_labels = torch.Tensor(self.test_labels)
        self.query_labels = torch.Tensor(self.query_labels)
        if cuda:
            self.test_labels = self.test_labels.cuda()
            self.query_labels = self.query_labels.cuda()

        self.dataloader_test = dataloader_test
        self.dataloader_query = dataloader_query
        self.cuda = cuda

    def ranks_map(self, model, remove_fc=False, features_normalized=True):
        test_descriptors = self.descriptors(self.dataloader_test, model)
        # query_descriptors = self.descriptors(self.dataloader_query, model)
        query_descriptors = test_descriptors

        square_features1 = torch.sum(query_descriptors * query_descriptors, dim=1)
        square_features2 = torch.sum(test_descriptors * test_descriptors, dim=1)
        correlationTerm = torch.mm(query_descriptors, test_descriptors.transpose(0, 1))
        dists = square_features1.view(-1, 1) - 2 * correlationTerm + square_features2.view(1, -1)
        dists[dists < 0] = 0.0

        dists_sorted, dists_sorted_inds = torch.sort(dists)

        def sort_by_dists_inds(data):
            return torch.gather(data.repeat(self.query_labels.shape[0], 1), 1, dists_sorted_inds)

        test_sorted_labels = sort_by_dists_inds(self.test_labels)

        eq_inds = self.query_labels == test_sorted_labels
        nmi = 0.0
        for _ in range(5):
            label_pre = KMeans(n_clusters=len(np.unique(self.test_labels.cpu().numpy())),n_jobs=-1,precompute_distances=True).fit(test_descriptors.cpu().double().numpy()).labels_
            nmi += metrics.normalized_mutual_info_score(self.test_labels.cpu().numpy(),label_pre) * 1/5


        eq_inds = eq_inds[:,1:]

        eq_inds = eq_inds.float()
        ranks = torch.cumsum(eq_inds,dim=1)
        ranks[ranks > 1] = 1
        ranks = torch.sum(ranks,dim=0) / eq_inds.size()[0]
        eq_cumsum = torch.cumsum(eq_inds,dim=1) * eq_inds
        eq_sum = torch.sum(eq_inds,dim=1)
        ranking = torch.arange(1, eq_inds.size()[1] + 1).view(1,-1).cuda().float()
        mAP = torch.mean(torch.sum(eq_cumsum / ranking, dim = 1) / eq_sum)

        return ranks, mAP, nmi

    def descriptors(self, dataloder, model):
        result = torch.FloatTensor()
        if self.cuda:
            result = result.cuda()
        for data in dataloder:
            if self.cuda:
                data = data.cuda()
            inputs = Variable(data)
            outputs = model(inputs)
            result = torch.cat((result, outputs.data), 0)

        return result
