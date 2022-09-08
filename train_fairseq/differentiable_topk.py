import torch
import torch.nn.functional as F

def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    
    v = torch.ones([bs, 1, k_]) / ( k_ ) # [bs, 1, k+1] 全て1./(k+1)
    G = torch.exp( -C / epsilon )
    if torch.cuda.is_available():
        v = v.cuda()
    
    for i in range(max_iter):
        u = mu / (G*v).sum( -1, keepdim=True )
        v = nu / (G*u).sum( -2, keepdim=True )
    
    Gamma = u*G*v
    return Gamma

def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    k = k_-1
    
    f = torch.zeros([bs, n, 1])
    g = torch.zeros([bs, 1, k+1])
    if torch.cuda.is_available():
        f = f.cuda()
        g = g.cuda()
    
    epsilon_log_mu = epsilon*torch.log(mu)
    epsilon_log_nu = epsilon*torch.log(nu)
    
    def min_epsilon_row(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -1, keepdim=True)
    
    def min_epsilon_col(Z, epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon, -2, keepdim=True)
    
    for i in range(max_iter):
        f = min_epsilon_row(C-g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C-f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma

def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    """
    Input
        grad_output_Gamma: Is its shape the same as Gamma?
        Gamma: shape:[bs, n, k+1]
        mu: shape:[1, n, 1], all values are 1./n
        nu: shape:[1, 1, k+1]
    Output:
        grad_output_C: shape:[bs, n, k+1]
    """

    nu_ = nu[:,:,:-1]
    Gamma_ = Gamma[:,:,:-1]
    
    bs, n, k_ = Gamma.size()
    
    inv_mu = 1./(mu.view([1, -1])) # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_) # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa+1e-8) # [bs, k, k] # Avoiding zero division error by adding 1e-8.
    
    Gamma_mu = inv_mu.unsqueeze(-1)*Gamma_
    L = Gamma_mu.matmul(inv_Kappa) # [bs, n, k]
    G1 = grad_output_Gamma * Gamma # [bs, n, k+1]
    
    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L) # [bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma # [bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode="constant", value=0) * Gamma # [bs, n, k+1]
    G2 = G21 + G22 + G23 # [bs, n, k+1]
    
    del g1, G21, G22, G23, Gamma_mu
    
    g2 = G1.sum(-2).unsqueeze(-1) # [bs, k+1, 1]
    g2 = g2[:,:-1,:] # [bs, k, 1]
    G31 = - L.matmul(g2) * Gamma # [bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode="constant", value=0) * Gamma # [bs, n, k+1]
    G3 = G31 + G32 # [bs, n, k+1]
    
    grad_C = (-G1+G2+G3)/epsilon # [bs, n, k+1]
    return grad_C

class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):
        """
        Input:
            C: [bs, n, k+1]
            mu: [1, n, 1] 全ての値が1./n
            nu: [1, 1, k+1] [:,:,:k]の値が1./n、[:,:,k]の値が(n-k)/n
        Output:
            Gamma: [bs, n, k+1]
        """
        
        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma!=Gamma)):
                    print("Nan appeared in Gamma, re-computing...")
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma
    
    @staticmethod
    def backward(ctx, grad_output_Gamma):
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu: [1, n, 1]
        # nu: [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None

class TopK_custom(torch.nn.Module):
    def __init__(self, epsilon=0.001, max_iter=200):
        super(TopK_custom, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, scores, k, epsilon=None):
        self.anchors = torch.FloatTensor([0,1]).view([1, 1, 2])
        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

        if epsilon is not None: # if you changed q_length while forwarding
            self.epsilon = epsilon

        bs, n = scores.size()
        scores = scores.view([bs, n, 1])
        
        # find the -inf value and replate it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_==float("-inf")] = float("inf")
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores==float("-inf")
        scores = scores.masked_fill(mask, filled_value)
        
        C = (scores-self.anchors)**2 # [bs, n, 1] -> [bs, n, 2]
        C = C / (C.max().detach()) # Cの最大値を定数化して正規化している()
        
        mu = torch.ones([1, n, 1], requires_grad=False)/n
        nu = torch.FloatTensor([k/n, (n-k)/n]).view([1, 1, 2]) # [k-n, (n-k)/n]
        
        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()
        
        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)
        
        A = Gamma[:,:,:1]*n # 上位k個に選ばれなかったところは削る
        
        return A, None

class SortedTopK_custom(torch.nn.Module):
    def __init__(self, epsilon=0.001, max_iter=200):
        super(SortedTopK_custom, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, scores, k, epsilon=None):
        self.anchors = torch.FloatTensor([k-i for i in range(k+1)]).view([1, 1, k+1])
        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

        if epsilon is not None: # if you changed q_length while forwarding
            self.epsilon = epsilon

        bs, n = scores.size()
        scores = scores.view([bs, n, 1])
        
        # find the -inf value and replate it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_==float("-inf")] = float("inf")
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores==float("-inf")
        scores = scores.masked_fill(mask, filled_value)
        
        C = (scores-self.anchors)**2 # [bs, n, 1] -> [bs, n, k+1]
        C = C / (C.max().detach()) # Cの最大値を定数化して正規化している()
        
        mu = torch.ones([1, n, 1], requires_grad=False)/n
        nu = [1./n for _ in range(k)]
        nu.append((n-k)/n)
        nu = torch.FloatTensor(nu).view([1, 1, k+1]) # 上位k個とそれ以外でk+1個
        
        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()
        
        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)
        
        A = Gamma[:,:,:k]*n # 上位k個に選ばれなかったところは削る
        
        return A, None