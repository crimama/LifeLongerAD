import torch 
import torch.nn as nn 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from models import PatchCore
import math 
import torch.nn as nn 

from .criteria import ArcMarginProduct

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP,self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size 
        self._output_size = output_size 
        self.net = self.__net__(input_size, hidden_size, output_size)
        
    def __net__(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(in_features= input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        
    def forward(self, inputs):
        return self.net(inputs) 
    
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features 
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = idx - (784 * (idx//784))
        return x,y 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class CoreProxy(nn.Module):
    def __init__(self,
                 init_core :np.ndarray,
                 temperature=0.05,
                 loss_fn = 'CrossEntropy'):
        '''
        default tempearture : 0.05
        '''
        super(CoreProxy, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(init_core))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = loss_fn

    def forward(self, embeddings, instance_targets, reduction:bool=True) -> torch.Tensor:
        norm_weight = nn.functional.normalize(self.weight, dim=1).to(embeddings.device)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        
        if self.loss_fn == 'CrossEntropy':
            loss = nn.CrossEntropyLoss()(prediction_logits / self.temperature, instance_targets)
        elif self.loss_fn == 'focalloss':
            loss = FocalLoss(reduce = reduction)(prediction_logits / self.temperature, instance_targets.to(prediction_logits.device))
        else:
            raise NotImplementedError
                
        return loss
    
class ProxyAnchor(torch.nn.Module):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    def __init__(self, 
                 init_core :np.ndarray,
                 scale = 32, 
                 margin = 0.1
                 ):
        super(ProxyAnchor, self).__init__()
        self.proxy = nn.Parameter(torch.Tensor(init_core))
        self.n_classes = self.proxy.shape[0]
        self.alpha = scale
        self.delta = margin

    def forward(self, embeddings, target):
        embeddings_l2 = F.normalize(embeddings, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=0)
        
        # N, dim, cls

        sim_mat = embeddings_l2.matmul(proxy_l2.T) # (N, cls)
        
        pos_target = F.one_hot(target, self.n_classes).float()
        neg_target = 1.0 - pos_target
        
        pos_mat = torch.exp(-self.alpha * (sim_mat - self.delta)) * pos_target
        neg_mat = torch.exp(self.alpha * (sim_mat + self.delta)) * neg_target
        
        pos_term = 1.0 / torch.unique(target).shape[0] * torch.sum(torch.log(1.0 + torch.sum(pos_mat, axis=0)))
        neg_term = 1.0 / self.n_classes * torch.sum(torch.log(1.0 + torch.sum(neg_mat, axis=0)))

        loss = pos_term + neg_term

        return loss
    
class ProxyNCA(torch.nn.Module):
    def __init__(self, 
        init_core :np.ndarray,
        smoothing_const = 0.1,
        scaling_x = 1,
        scaling_p = 3
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = nn.Parameter(torch.Tensor(init_core))
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T, mean=True, reduction:bool=None):
        def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
            # Optional: BNInception uses label smoothing, apply it for retraining also
            # "Rethinking the Inception Architecture for Computer Vision", p. 6
            T = torch.eye(nb_classes).to(T.device)[T]
            T = T * (1 - smoothing_const)
            T[T == 0] = smoothing_const / (nb_classes - 1)
            return T
        '''
        learning using initial position based fixed label 
        '''
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        if mean:
            return loss.mean()
        else:
            return loss
        
class SoftTriple(nn.Module):
    def __init__(self, init_core:np.ndarray, la=20, gamma=0.1, tau=0.2, margin=0.01, dim=1024, K=4):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = init_core.shape[0]
        self.K = K
        self.fc = nn.Parameter(torch.Tensor(init_core).repeat(K,1).T)
        self.weight = torch.zeros(self.cN*K, self.cN*K, dtype=torch.bool).cuda()
        for i in range(0, self.cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1

    def forward(self, input, target):
        device = input.device
        centers = F.normalize(self.fc, p=2, dim=0).to(device)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target.to(device))
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify    


class ProxyCore(nn.Module):
    '''
    pslabel_sampling_ratio 는 fit에서 초기 pseudo label 생성을 위한 coreset을 위한 용도
    당장 여기서 사용 되지는 않지만 config yaml파일에서 params.로 넣기 위해 임시로 넣어둠 
    '''
    def __init__(self, backbone, faiss_on_gpu, faiss_num_workers, pslabel_sampling_ratio,
                 sampling_ratio, device, input_shape, threshold='quant_0.15', weight_method='identity',
                 n_input_feat:int=1024, n_hidden_feat:int=4096, n_projection_feat:int=1024,
                 temperature=0.05, loss_fn='CrossEntropy', proxy='CoreProxy'
                ):
        super(ProxyCore,self).__init__()
        self.core = PatchCore(
            backbone          = backbone,
            faiss_on_gpu      = faiss_on_gpu,
            faiss_num_workers = faiss_num_workers,
            sampling_ratio    = sampling_ratio,
            device            = device,
            input_shape       = input_shape,
            threshold         = threshold,
            weight_method     = weight_method
            
        )
        
        self.embedding_layer = MLP(n_input_feat,n_hidden_feat,n_projection_feat)
        self.projection_layer = MLP(n_projection_feat,n_hidden_feat,n_projection_feat)
        
        
        self.device = device
        self.temperature = temperature
        self.loss_fn = loss_fn
        self.proxy = proxy 
        
    def set_criterion(self, init_core):
        if self.proxy == 'CoreProxy':
            self._criterion = CoreProxy(
            init_core   = init_core,
            temperature = self.temperature,
            loss_fn     = self.loss_fn
            )
        elif self.proxy == 'ProxyNCA':
            self._criterion = ProxyNCA(
                *init_core.shape
            )
        elif self.proxy == 'ProxyAnchor':
            self._criterion = ProxyAnchor(
                init_core = init_core
            )
            
        elif self.proxy == 'SoftTriple':
            self._criterion = SoftTriple(
                init_core = init_core
            )
        elif self.proxy == 'ArcMarginProduct':
            self._criterion = ArcMarginProduct(
                init_core = init_core
            )
                    
        
    def criterion(self, outputs:list):
        '''
            outputs = [z,w]
        '''
        return self._criterion(*outputs)
    
    def get_patch_embed(self, trainloader):
        self.core.forward_modules.eval()
        
        features = [] 
        for imgs, labels, gts in trainloader:
            imgs = imgs.to(torch.float).to(self.device)
            with torch.no_grad():
                feature = self.core._embed(imgs) # (N*P*P,C)
                feature = torch.Tensor(np.vstack(feature))
                features.append(feature)        
            
        origin_patch_embeds = torch.concat(features)
        return origin_patch_embeds

    def get_feature_loader(self, trainloader, labels=None):
        features = self.get_patch_embed(trainloader)
        featuredataset = FeatureDataset(features, labels=labels)
        featureloader = DataLoader(featuredataset, batch_size=2048, shuffle=True)
        return featureloader
        
    def forward(self, feat):     
        output = self.projection_layer(self.embedding_layer(feat))   
        return output   
        
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels=None):
        self.features = features 
        self.labels = labels 
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = idx - (784 * (idx//784)) if self.labels is None else self.labels[idx]        
        return x,y 