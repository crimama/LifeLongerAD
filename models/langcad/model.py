from .sampler import ApproximateGreedyCoresetSampler
from .common import PatchMaker, NearestNeighbourScorer, RescaleSegmentor, FaissNN
from sklearn.metrics.pairwise import cosine_similarity
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import timm 
import open_clip 
import numpy as np 


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)    

class POOL:
    def __init__(self):
        self.key = {}
        self.knowledge = [] 
        self.prompts = []
    def save_pool(self, save_path:str):
        pool_state = {'prompts':self.prompts, 'knowledge':self.knowledge}
        torch.save(pool_state,save_path)
        print('Save done')
        
    def load_pool(self, load_path:str):
        pool_state = torch.load(load_path)
        self.prompts, self.knowledge = pool_state['prompts'], pool_state['knowledge']
        print('Load done')

        
    def get_knowledge(self, class_name:str=None, knowledge=None):
        # key save 
        self.key[class_name] = np.mean(knowledge,axis=0)
        self.knowledge.extend(knowledge)
        return self.knowledge

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
    
    def retrieve_prompts(self, prompts, query_features:np.ndarray):
        '''
            저장해 있는 prompts 파라미터들 중 top k 선택 후 
            input으로 들어온 prompt의 파라미터를 바꿔준 뒤 return 
            일단 feature block은 3 한개로 고정하는 것으로 진행 
        '''
        key_prompt = np.vstack(self.prompts) # pool 에 저장되어 있는 prompts 
        
        similarities = cosine_similarity(
            query_features.reshape(1,-1),
            key_prompt
        )
        k = prompts.num_prompts  
        top_k_indices = np.argsort(similarities)[0][-k:][::-1]  # 유사도가 높은 순으로 정렬
        prompts_parms = [self.prompts[i] for i in top_k_indices]
        num_layer = prompts.num_layers[0]
        prompts.prompts[str(num_layer)] = nn.Parameter(torch.Tensor(prompts_parms)) # key : block 번호 
        return prompts 
    
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
        
        num_prompts: int = 6,
        prompt_dim: int = 768,
        prompt_method: str = 'intermediate', # intermediate, input, null 
        txt_emb_method =  None, # None, int 
        pre_task = 'margin' # margin, infonce
    ):
        super(LANGCAD,self).__init__()
        
        #! timm ViT 
        if backbone['pretrained'] != 'openai':
            self.encoders, self.preprocess_val, self.preprocess = open_clip.create_model_and_transforms(backbone['name'])
            self.tokenizer = open_clip.get_tokenizer(backbone['name'])
            self.encoders.visual = timm.create_model(backbone['pretrained'], pretrained=True)
        else:
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
        self.criterion = ClipLoss(pre_task = pre_task)
        
        self.device         = device 
        self.num_layers     = num_layers if type(num_layers)==list else [num_layers]
        self.num_prompts    = num_prompts 
        self.prompt_dim     = prompt_dim 
        self.prompt_method  = prompt_method
        self.txt_emb_method = txt_emb_method        
        
        for param in self.parameters():
            param.requires_grad = False 
        
    def create_prompts(self):
        return Prompts(num_layers = self.num_layers, num_prompts = self.num_prompts, prompt_dim = self.prompt_dim)
    
    #! last embedding 사용 
    def embed_text(self, text:str, txt_emb_method:int=None) -> torch.Tensor:
        '''
            txt_emb_method = [None, int]
            None : Last 
            int : n intermediate layer
        '''
        with torch.no_grad():
            text = self.tokenizer(text).to(self.device)

            cast_dtype = self.encoders.transformer.get_cast_dtype()

            x = self.encoders.token_embedding(text).to(cast_dtype)
            x = x + self.encoders.positional_embedding.to(cast_dtype)
            
            if txt_emb_method is None:
                x - self.encoders.transformer(x, attn_mask=self.encoders.attn_mask)
            else:
                for i, blk in enumerate(self.encoders.transformer.resblocks):
                    x = blk(x, attn_mask=self.encoders.attn_mask)
                    if i == txt_emb_method-1:  # 3번째 resblocks에서의 feature를 return
                        break
                x = self.encoders.ln_final(x)                    
        return x
    
    #! open clip 
    def embed_img(self, img, prompts=None):
        # Patch embedding 단계에서 CLS 토큰 추가 후 prompt 추가
        x = self.encoders.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # CLS 토큰 추가
        x = torch.cat([_expand_token(self.encoders.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        x = x + self.encoders.visual.positional_embedding.to(x.dtype)
        x = self.encoders.visual.patch_dropout(x)
        x = self.encoders.visual.ln_pre(x)

        if self.prompt_method is not None: 
            # Prompt 추가 (input 방식일 경우 처음에 추가)
            if self.prompt_method == 'input' and prompts is not None:
                prompt = prompts[self.num_layers[0]].unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([x, prompt], dim=1)

        features = []
        for i, blk in enumerate(self.encoders.visual.transformer.resblocks):
            if self.prompt_method == 'intermediate' and prompts is not None and self.prompt_method is not None and i in self.num_layers:
                # Intermediate prompt 추가 방식
                prompt = prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                if i == self.num_layers[0]:
                    x = torch.cat([x, prompt], dim=1)
                else:
                    x = torch.cat([x[:, :-prompt.shape[1], :], prompt], dim=1)

            x = blk(x)
            if i in self.num_layers:
                if self.prompt_method is not None: 
                    # features.append(x[:,1:-self.num_prompts,:])
                    # features.append(x[:,-self.num_prompts:,:])                  
                    features.append(x)
                else: 
                    features.append(x)

            if i == self.num_layers[-1]:
                break

        # image_features = torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]
        image_features = torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]

        return image_features
    
    #! Timm ViT 
    # def embed_img(self, img, prompts=None):
    #     # Patch embedding 단계에서 CLS 토큰 추가 후 prompt 추가
    #     x = self.encoders.visual.patch_embed(img)
    #     x = self.encoders.visual._pos_embed(x)
    #     x = self.encoders.visual.pos_drop(x)
    #     x = self.encoders.visual.norm_pre(x)

    #     if self.prompt_method is not None: 
    #         # Prompt 추가 (input 방식일 경우 처음에 추가)
    #         if self.prompt_method == 'input' and prompts is not None:
    #             prompt = prompts[self.num_layers[0]].unsqueeze(0).expand(x.shape[0], -1, -1)
    #             x = torch.cat([x, prompt], dim=1)

    #     features = []
    #     for i, blk in enumerate(self.encoders.visual.blocks):
    #         if self.prompt_method == 'intermediate' and prompts is not None and self.prompt_method is not None and i in self.num_layers:
    #             # Intermediate prompt 추가 방식
    #             prompt = prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
    #             if i == self.num_layers[0]:
    #                 x = torch.cat([x, prompt], dim=1)
    #             else:
    #                 x = torch.cat([x[:, :-prompt.shape[1], :], prompt], dim=1)

    #         x = blk(x)
    #         if i in self.num_layers:
    #             if self.prompt_method is not None: 
    #                 features.append(x[:,1:-self.num_prompts,:])
    #             else: 
    #                 features.append(x[:,1:,:])

    #         if i == self.num_layers[-1]:
    #             break

    #     # image_features = torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]
    #     image_features = torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)

    #     return image_features       
    
    #! def forward(self, img:torch.Tensor, positive:list, prompts=None):
    def forward(self, img:torch.Tensor, positive:list, negative:list, prompts=None):
        visual_features = self.embed_img(img, prompts)
        visual_features = visual_features.mean(dim=1)@self.proj
        
        # text_features = self.embed_text(positive)
        # text_features = text_features.mean(dim=1)
        
        pos_text_features = self.embed_text(positive, txt_emb_method=self.txt_emb_method)
        pos_text_features = pos_text_features.mean(dim=1)
        
        
        neg_text_features = torch.cat([self.embed_text(t, txt_emb_method=self.txt_emb_method) for t in negative])        
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
            img_features = self.embed_img(img, prompts)[:,:-prompts.num_prompts,:]
        else:
            img_features = self.embed_img(img)
            
        img_features = np.asarray(img_features.detach().cpu()).reshape(-1,img_features.shape[-1]).astype(np.float32)        
        
        score, _, indicies = self.anomaly_scorer.predict([img_features])
        
        img_score = self.patchmaker.unpatch_scores(score,shape[0])
        img_score = self.patchmaker.score(img_score)
        
        pixel_score = self.patchmaker.unpatch_scores(score,shape[0])
        pixel_score = pixel_score.reshape(shape[0],int(pixel_score.shape[1]**0.5),-1)
        pixel_score = np.array(self.anomaly_segmentor.convert_to_segmentation(pixel_score))
                
        return score, img_score, pixel_score
    
class ClipLoss(nn.Module):
    def __init__(self,
                 pre_task):
        super().__init__()

        self.prev_num_logits = 0
        self.labels = {}
        self.pre_task = pre_task # margin, infonce, None 

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, neg_text_features, logit_scale=1/0.07) -> dict:
        device = image_features.device
        
        if self.pre_task == 'margin' or self.pre_task is None:
            
            #! Contrastive loss 
            # logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
            # labels = self.get_ground_truth(device, logits_per_image.shape[0])
            # breakpoint()
            # contrastive_loss = (
            #     F.cross_entropy(logits_per_image, labels) +
            #     F.cross_entropy(logits_per_text, labels)
            # ) / 2
            contrastive_loss = torch.Tensor([0]).to(device)
            
            #! cosine sim loss 
            # l2_distance = torch.norm(image_features - text_features, p=2, dim=1)
            # contrastive_loss = torch.mean(l2_distance)            
            
            # Negative logits (image and negative text)
            if self.pre_task == 'margin':
                neg_logits_per_image = logit_scale * image_features @ neg_text_features.T
                neg_labels = torch.full((neg_logits_per_image.shape[1],), -1, device=device, dtype=torch.long)
                negative_loss = F.margin_ranking_loss(neg_logits_per_image, torch.zeros_like(neg_logits_per_image), neg_labels, margin=0.1)
                total_loss = contrastive_loss + negative_loss
                
            else:
                total_loss = contrastive_loss 
                negative_loss =  torch.Tensor([0])
            
        elif self.pre_task == 'infonce': 
            text_features = torch.cat([text_features, neg_text_features])        
            logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            
            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text[:len(labels)], labels)
            ) / 2            
            
            total_loss = contrastive_loss
            negative_loss = torch.Tensor([0])
        return {'total_loss':total_loss,'contrastive_loss':contrastive_loss,'negative_loss':negative_loss} 

    
class Prompts(nn.Module):
    def __init__(self, num_layers: list = [3], num_prompts: int = 3, prompt_dim: int = 768):
        super(Prompts, self).__init__()
        self.prompts = nn.ParameterDict({
            str(block): nn.Parameter(torch.nn.init.uniform_(torch.randn(num_prompts, prompt_dim), -1, 1), requires_grad=True)
            for block in num_layers
        })
        self.num_prompts = num_prompts
        self.num_layers = num_layers 
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, key):
        return self.prompts[str(key)]
    