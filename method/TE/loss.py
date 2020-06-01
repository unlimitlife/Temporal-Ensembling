import torch
import torch.nn.functional as F
from torch.autograd import Variable


def temporal_loss(out1, prediction, w, labels):
    
    # MSE between current and temporal outputs
    def mse_loss(out1, prediction):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - prediction) ** 2)
        return quad_diff / out1.data.nelement()
    
    def masked_crossentropy(out, labels):
        cond = (labels >= 0)
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs, masked_labels)
            #loss = F.nll_loss(torch.log(masked_outputs), masked_labels)
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0
    
    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, prediction)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup