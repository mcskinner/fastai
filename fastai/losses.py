import torch

def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9):
    def safe_div(x, y): return x/(y+eps)
    beta2 = beta**2
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)  # true positives
    precision = safe_div(tp, y_pred.sum(dim=1), eps)
    recall = safe_div(tp, y_true.sum(dim=1), eps)
    return torch.mean(save_div((1+beta2)*precision*recall, beta2*precision+recall, eps)
