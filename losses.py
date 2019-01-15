import torch
from torch.autograd import Variable
def pairwise_distance(features1,features2,squared=False):
    square_features1 = torch.sum(features1 * features1, dim=1)
    square_features2 = torch.sum(features2 * features2, dim=1)
    dot_product = torch.mm(features1, features2.transpose(0, 1))
    square_dists = square_features1.view(-1, 1) - 2 * dot_product + square_features2.view(1, -1)
    if squared:
        square_dists[square_dists < 0.0] = 0.0
        dist = square_dists
    else:
        square_dists[square_dists < 1e-16] = 1e-16 #prevent nan.
        dist = torch.sqrt(square_dists)
    return dist
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin = 1.0, cuda=True):
        super(ContrastiveLoss, self).__init__()
        self.cuda = cuda
        self.margin = margin

    def forward(self, features, labels):
        labels = labels.view(-1)
        positive_mask = labels.view(1, -1) == labels.view(-1, 1)
        square_dists = pairwise_distance(features,features, squared=True)
        square_dists = torch.triu(square_dists,1)
        if self.cuda:
            n_dists = torch.max(torch.zeros(square_dists.size()).cuda(), (self.margin - square_dists))
        else:
            n_dists = torch.max(torch.zeros(square_dists.size()), (self.margin - square_dists))
        p_pairs = square_dists[positive_mask]
        n_pairs = n_dists[~positive_mask]
        pair_num = torch.sum(p_pairs > 1e-16).item() + torch.sum(n_pairs > 1e-16).item()
        loss = (torch.sum(square_dists[positive_mask]) + torch.sum(n_dists[~positive_mask])) / pair_num
        return loss

class TripletLoss(torch.nn.Module):
    def __init__(self, margin = 1.0, cuda=True, mining='hardest'):
        super(TripletLoss, self).__init__()
        self.cuda = cuda
        self.margin = margin
        self.mining = mining
    def forward(self, features, labels):
        labels = labels.view(-1)
        batch_size = labels.size()[0]
        positive_mask = labels.view(1, -1) == labels.view(-1, 1)
        negative_mask = ~positive_mask
        square_dists = pairwise_distance(features,features, squared=True)
        if self.mining == 'hardest':
            square_dists_repeated = square_dists.repeat(1, batch_size).view(-1, batch_size)
            all_positive = square_dists.view(-1, 1)
            positive_mask_repeated = positive_mask.repeat(1, batch_size).view(-1, batch_size)
            square_dists_repeated[positive_mask_repeated] = float("inf")
            hardest_negatives = torch.min(square_dists_repeated,1,keepdim=True)[0]
            if self.cuda:
                loss = torch.sum(torch.max(torch.zeros(all_positive.size()).cuda(), all_positive - hardest_negatives + self.margin)[positive_mask.view(-1,1)]) / positive_mask.sum().item()
            else:
                loss = torch.sum(torch.max(torch.zeros(all_positive.size()), all_positive - hardest_negatives + self.margin)[positive_mask.view(-1,1)]) / positive_mask.sum().item()
            return loss
        if self.mining == 'semi':
            square_dists_repeated = square_dists.repeat(1,batch_size).view(-1,batch_size)
            all_positive = square_dists.view(-1, 1)
            positive_mask_repeated = positive_mask.repeat(1, batch_size).view(-1, batch_size)
            diff_dist = square_dists_repeated - all_positive
            diff_dist[positive_mask_repeated] = 0
            larger_inds = torch.sum(diff_dist > 1e-16, 1, keepdim=True) >= 1
            smaller_inds = ~larger_inds
            larger_dist = (diff_dist > 1e-16).float() * diff_dist
            smaller_dist = (diff_dist < -(1e-16)).float() * diff_dist
            larger_dist[larger_dist <= 1e-16] = float('inf')
            smaller_dist[smaller_dist >= -(1e-16)] = float('-inf')
            semi_hard_ind_larger = torch.min(larger_dist, 1, keepdim=True)[1]
            semi_hard_ind_smaller = torch.max(smaller_dist, 1, keepdim=True)[1]
            semi_hard_ind = semi_hard_ind_larger * larger_inds.long() + semi_hard_ind_smaller * smaller_inds.long()
    #-------------------------------------------------------------------
            semi_negative = square_dists_repeated.gather(1,semi_hard_ind)
            if self.cuda:
                loss_matrix = torch.max(torch.zeros(all_positive.size()).cuda(), all_positive - semi_negative + self.margin).view(batch_size,batch_size)
            else:
                loss_matrix = torch.max(torch.zeros(all_positive.size()), all_positive - semi_negative + self.margin).view(batch_size,batch_size)
            loss = torch.sum(loss_matrix[positive_mask]) / positive_mask.sum().item()
            return loss
        if self.mining == 'all':
            square_dists_repeated = square_dists.repeat(1,batch_size).view(-1,batch_size)
            negative_mask_repeated = negative_mask.repeat(1, batch_size).view(-1, batch_size)
            inds_valid = positive_mask.view(-1,1) * negative_mask_repeated
            if self.cuda:
                loss_valid = torch.max(torch.zeros(square_dists_repeated.size()).cuda(), square_dists.view(-1, 1) - square_dists_repeated + self.margin)[inds_valid]
            else:
                loss_valid = torch.max(torch.zeros(square_dists_repeated.size()), square_dists.view(-1, 1) - square_dists_repeated + self.margin)[inds_valid]
            loss = torch.sum(loss_valid) / inds_valid.sum().item()
            return loss

class LiftedLoss(torch.nn.Module):
    def __init__(self, margin = 1.0, cuda=True):
        super(LiftedLoss, self).__init__()
        self.cuda = cuda
        self.margin = margin
    def forward(self, features, labels):
        labels = labels.view(-1)
        batch_size = labels.size()[0]
        positive_mask = labels.view(1, -1) == labels.view(-1, 1)
        negative_mask = ~positive_mask
        dists = pairwise_distance(features, features, squared=False)
    #-------------loss--------------------------
        dists_repeated_x = dists.repeat(1,batch_size).view(-1,batch_size)
        negative_mask_repeated_x = negative_mask.repeat(1,batch_size).view(-1,batch_size).float()
        dists_repeated_y = dists.transpose(1,0).repeat(batch_size,1)
        negative_mask_repeated_y = negative_mask.transpose(1,0).repeat(batch_size,1).float()
        positive_dists = dists.view(-1, 1)
        J_matrix = torch.log(torch.sum(torch.exp(self.margin - dists_repeated_x) * negative_mask_repeated_x,1,keepdim=True) + torch.sum(torch.exp(self.margin - dists_repeated_y) * negative_mask_repeated_y,1,keepdim=True)) + positive_dists
        J_matrix_valid = torch.masked_select(J_matrix, positive_mask.view(-1, 1))
        if self.cuda:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()).cuda(), J_matrix_valid)
        else:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()), J_matrix_valid)
        lifted_loss_matrix = J_matrix_valid * J_matrix_valid
        lifted_loss = torch.sum(lifted_loss_matrix) / (2 * positive_mask.sum().item())
        return lifted_loss

class ProxyNCA(torch.nn.Module):
    def __init__(self, embeding_size, classes_num, batchsize, cuda=True):
        super().__init__()
        self.cuda = cuda
        self.embeding_size = embeding_size
        self.classes_num = classes_num
        self.batchsize = batchsize
        self.proxies = Variable(torch.rand(classes_num, embeding_size))
        if self.cuda:
            self.proxies = self.proxies.cuda()
        torch.nn.init.xavier_uniform_(self.proxies)
        self.proxies_labels = torch.arange(0, self.classes_num).long().view(1,-1)
        if self.cuda:
            self.proxies_labels = self.proxies_labels.cuda()

    def forward(self, features, labels):
        labels_eq = labels.view(-1,1).long() == self.proxies_labels
        normed_proxies = torch.nn.functional.normalize(self.proxies, dim=1)
        normed_features = torch.nn.functional.normalize(features, dim=1)
        square_dists = pairwise_distance(normed_features, normed_proxies, squared=True)
        p_dists = square_dists[labels_eq].view(-1,1)
        n_dists = square_dists[~labels_eq].view(self.batchsize, self.classes_num - 1)
        loss = -torch.log(torch.exp(-p_dists) / torch.sum(torch.exp(-n_dists), dim=1, keepdim=True))
        loss = torch.mean(loss)
        return loss