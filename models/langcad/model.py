from .sampler import ApproximateGreedyCoresetSampler
from .common import PatchMaker, NearestNeighbourScorer, RescaleSegmentor, FaissNN
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import open_clip 
import numpy as np 


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)    

class POOL:
    def __init__(self):
        self.key = {} 
        self.knowledge = {} 
        self.prompts = {}

        
    def get_knowledge(self, class_name:str=None, knowledge=None):
        # key save 
        self.key[class_name] = np.mean(knowledge,axis=0)
        self.knowledge[class_name] = knowledge

    def retrieve_key(self, model, testloader):
        query = [] 
        for imgs, labels, gts in testloader:
            features = model.embed(imgs).detach().cpu().numpy()
            query.append(features)
        
        query = np.concatenate(query).mean(axis=(0,1))
        key = np.vstack(model.pool.key.values())
        
        class_index = np.argmin(np.linalg.norm(query-key,axis=-1))    
        class_name = list(self.key.keys())[class_index]
        return class_name 
        
class LANGCAD(nn.Module):
    def __init__(
        self,
        backbone: dict, #clip model
        input_shape: list,
        num_layers: list = [3],
        device: str = 'cuda',
        faiss_on_gpu: bool = False, 
        faiss_num_workers: int = 4,
        sampling_ratio: float = 0.1,
        n_neighbors: int = 5,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        num_prompts: int = 3,
        prompt_dim: int = 768,
    ):
        super(LANGCAD,self).__init__()
        
        self.encoders, self.preprocess_val, self.preprocess = open_clip.create_model_and_transforms(backbone['name'], pretrained=backbone['pretrained'])
        self.tokenizer = open_clip.get_tokenizer(backbone['name'])
        
        self.proj = self.encoders.visual.proj
        self.patchmaker = PatchMaker()
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=n_neighbors, nn_method=FaissNN(on_gpu = faiss_on_gpu, num_workers = faiss_num_workers)
            )
        self.anomaly_segmentor = RescaleSegmentor(
            device=device, target_size=input_shape[-2:]
            )
        self.featuresampler = ApproximateGreedyCoresetSampler(
            percentage=sampling_ratio, device=device
            )
        
        self.pool = POOL()
        self.criterion = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels
        )
        
        self.device         = device 
        self.num_layers     = num_layers
        self.num_prompts    = num_prompts 
        self.prompt_dim     = prompt_dim 
        
        for param in self.parameters():
            param.requires_grad = False 

        
    def create_prompts(self):
        return Prompts(num_blocks = self.num_layers, num_prompts = self.num_prompts, prompt_dim = self.prompt_dim)
    
    #! last embedding 사용 
    def _embed_text(self, text:str) -> torch.Tensor:
        with torch.no_grad():
            text = self.tokenizer(text).to(self.device)

            cast_dtype = self.encoders.transformer.get_cast_dtype()

            x = self.encoders.token_embedding(text).to(cast_dtype)
            x = x + self.encoders.positional_embedding.to(cast_dtype)
            x - self.encoders.transformer(x, attn_mask=self.encoders.attn_mask)
            x = self.encoders.ln_final(x)                    
        return x
    
    #! 중간 block에서 빼기 
    # def _embed_text(self, text: str) -> torch.Tensor:
    #     with torch.no_grad():
    #         text = self.tokenizer(text).to(self.device)

    #         cast_dtype = self.encoders.transformer.get_cast_dtype()

    #         x = self.encoders.token_embedding(text).to(cast_dtype)
    #         x = x + self.encoders.positional_embedding.to(cast_dtype)
    #         for i, blk in enumerate(self.encoders.transformer.resblocks):
    #             x = blk(x, attn_mask=self.encoders.attn_mask)
    #             if i == 2:  # 3번째 resblocks에서의 feature를 return
    #                 break
    #         x = self.encoders.ln_final(x)
    #     return x

    #! prompt 중간에 끼워 넣는 방식 
    def _embed_img(self, img, prompts = None):
        x = self.encoders.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([_expand_token(self.encoders.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        x = x + self.encoders.visual.positional_embedding.to(x.dtype)
        x = self.encoders.visual.patch_dropout(x)
        x = self.encoders.visual.ln_pre(x)

        features = []
        for i, blk in enumerate(self.encoders.visual.transformer.resblocks):
            if i in self.num_layers:
                if prompts is not None:
                    prompt = prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                    
                    if i == self.num_layers[0]:
                        x = torch.cat([x,prompt],dim=1)
                    else:
                        x = torch.cat([
                            x[:, :-prompt.shape[1], :], 
                            prompt
                            ], dim=1)
            x = blk(x)
            if i in self.num_layers:
                features.append(x)
            
            if i == self.num_layers[-1]:
                break 

        return torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]
    
    #! prompt를 input embedding에 넣고 계속 inference 
    # def _embed_img(self, img, prompts=None):
    #     # Patch embedding 단계에서 CLS 토큰 추가 후 prompt 추가
    #     x = self.encoders.visual.conv1(img)
    #     x = x.reshape(x.shape[0], x.shape[1], -1)
    #     x = x.permute(0, 2, 1)

    #     # CLS 토큰 추가
    #     x = torch.cat([_expand_token(self.encoders.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)


    #     x = x + self.encoders.visual.positional_embedding.to(x.dtype)
    #     x = self.encoders.visual.patch_dropout(x)
    #     x = self.encoders.visual.ln_pre(x)
        
    #     # Prompt 추가
    #     if prompts is not None:
    #         prompt = prompts[self.num_layers[0]].unsqueeze(0).expand(x.shape[0], -1, -1) #Todo 추후 다중 블록 사용 시 수정 필요 
    #         x = torch.cat([x, prompt], dim=1)

    #     features = []
    #     for i, blk in enumerate(self.encoders.visual.transformer.resblocks):
    #         x = blk(x)
    #         if i in self.num_layers:
    #             features.append(x)
    #         if i == self.num_layers[-1]:
    #             break

    #     return torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]
        
    
    def embed(self, img:torch.Tensor, prompts=None, out_with_prompts:bool=True) -> torch.Tensor:        
        image_features = self._embed_img(img, prompts)
        if prompts is not None:
            if out_with_prompts:
                return image_features 
            else:
                return image_features[:,:-prompts.num_prompts,:]
        else:
            return image_features 
    
    #! def forward(self, img:torch.Tensor, positive:list, prompts=None):
    def forward(self, img:torch.Tensor, positive:list, negative:list, prompts=None):
        visual_features = self._embed_img(img, prompts)
        visual_features = visual_features.mean(dim=1)@self.proj
        
        # text_features = self._embed_text(positive)
        # text_features = text_features.mean(dim=1)
        
        pos_text_features = self._embed_text(positive)
        pos_text_features = pos_text_features.mean(dim=1)
        
        neg_text_features = self._embed_text(negative)
        neg_text_features = neg_text_features.mean(dim=1)
        
        # loss = self.criterion(visual_features, text_features)
        loss = self.criterion(visual_features, pos_text_features, neg_text_features)
        return loss 
    
    @torch.no_grad()
    def fit(self, img_features:np.ndarray):
        if isinstance(img_features,torch.Tensor):
            img_features = img_features.detach().cpu().numpy()
            
        D = img_features.shape[-1]
        img_features = img_features.reshape(-1,D).astype(np.float32) 
        sample_features, sample_indices = self.featuresampler.run(img_features)
        return sample_features

    @torch.no_grad()
    def predict(self, img:torch.Tensor, prompts=None):
        shape = img.shape
        if prompts is not None:
            img_features = self.embed(img, prompts, out_with_prompts=False)
        else:
            img_features = self.embed(img)
            
        img_features = np.asarray(img_features.detach().cpu()).reshape(-1,img_features.shape[-1]).astype(np.float32)        
        
        score, _, indicies = self.anomaly_scorer.predict([img_features])
        
        img_score = self.patchmaker.unpatch_scores(score,shape[0])
        img_score = self.patchmaker.score(img_score)
        
        pixel_score = self.patchmaker.unpatch_scores(score,shape[0])
        pixel_score = pixel_score.reshape(shape[0],int(pixel_score.shape[1]**0.5),-1)
        pixel_score = np.array(self.anomaly_segmentor.convert_to_segmentation(pixel_score))
                
        return score, img_score, pixel_score
    
class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, pos_text_features, neg_text_features, logit_scale=1/0.07, output_dict=False):
        device = image_features.device
        
        # Positive logits (image and matching positive text)
        logits_per_image, logits_per_text = self.get_logits(image_features, pos_text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        
        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        # Negative logits (image and negative text)
        neg_logits_per_image = logit_scale * image_features @ neg_text_features.T
        neg_labels = torch.full((neg_logits_per_image.shape[0],), -1, device=device, dtype=torch.long)
        
        negative_loss = F.margin_ranking_loss(
            neg_logits_per_image, torch.zeros_like(neg_logits_per_image), neg_labels, margin=0.1
        )
        
        total_loss = contrastive_loss + negative_loss

        return {"contrastive_loss": total_loss} if output_dict else total_loss


#! Baseline clip loss (kiie method 2)  
# class ClipLoss(nn.Module):
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels

#         self.prev_num_logits = 0
#         self.labels = {}

#     def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]
#         return labels

#     def get_logits(self, image_features, text_features, logit_scale):
#         logits_per_image = logit_scale * image_features @ text_features.T
#         logits_per_text = logit_scale * text_features @ image_features.T
        
#         return logits_per_image, logits_per_text

#     def forward(self, image_features, text_features, logit_scale=1/0.07, output_dict=False):
#         device = image_features.device
#         logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

#         labels = self.get_ground_truth(device, logits_per_image.shape[0])

#         total_loss = (
#             F.cross_entropy(logits_per_image, labels) +
#             F.cross_entropy(logits_per_text, labels)
#         ) / 2

#         return {"contrastive_loss": total_loss} if output_dict else total_loss
    
    
class Prompts(nn.Module):
    def __init__(self, num_blocks: list = [3], num_prompts: int = 3, prompt_dim: int = 768):
        super(Prompts, self).__init__()
        self.prompts = nn.ParameterDict({
            str(block): nn.Parameter(torch.nn.init.uniform_(torch.randn(num_prompts, prompt_dim), -1, 1), requires_grad=True)
            for block in num_blocks
        })
        self.num_prompts = num_prompts
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, key):
        return self.prompts[str(key)]
    

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)