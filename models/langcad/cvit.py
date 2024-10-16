from .sampler import ApproximateGreedyCoresetSampler
from .common import PatchMaker, NearestNeighbourScorer, RescaleSegmentor, FaissNN
import torch 
import torch.nn as nn 
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
        
class CViT(nn.Module):
    def __init__(
        self,
        backbone: dict, #clip model
        input_shape: list,
        num_layers: list = [6,9],
        device: str = 'cuda',
        faiss_on_gpu: bool = False, 
        faiss_num_workers: int = 4,
        sampling_ratio: float = 0.1,
        n_neighbors: int = 5,
    ):
        super(CViT,self).__init__()
        
        self.encoders, self.preprocess_val, self.preprocess = open_clip.create_model_and_transforms(backbone['name'], pretrained=backbone['pretrained'])
        self.tokenizer = open_clip.get_tokenizer(backbone['name'])
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
        
        self.img_size       = input_shape
        self.sampling_ratio = sampling_ratio
        self.device         = device 
        self.num_layers     = num_layers
        
    def _embed_text(self, text:str) -> torch.Tensor:
        with torch.no_grad():
            text = self.tokenizer([f"a photo of {text}"]).to(self.device)

            cast_dtype = self.encoders.transformer.get_cast_dtype()

            x = self.encoders.token_embedding(text).to(cast_dtype)
            x = x + self.encoders.positional_embedding.to(cast_dtype)
            x - self.encoders.transformer(x, attn_mask=self.encoders.attn_mask)
            x = self.encoders.ln_final(x)                    
        return x
    
    # def _embed_img(self, img):
    #     x = self.encoders.visual.conv1(img)
    #     x = x.reshape(x.shape[0],x.shape[1],-1)
    #     x = x.permute(0,2,1)
        
    #     x = torch.cat([_expand_token(self.encoders.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        
    #     x = x + self.encoders.visual.positional_embedding.to(x.dtype)
    #     x = self.encoders.visual.patch_dropout(x)
    #     x = self.encoders.visual.ln_pre(x)
        
    #     features = [] 
    #     def hook_fn(module, input, output):
    #         features.append(output)
                        
    #     for n_l in self.num_layers:                        
    #         self.encoders.visual.transformer.resblocks[n_l].register_forward_hook(hook_fn)
        
    #     x = self.encoders.visual.transformer(x)
        
    #     return torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:,1:,:]
    def _embed_img(self, img):
        x = self.encoders.visual.conv1(img)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([_expand_token(self.encoders.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        x = x + self.encoders.visual.positional_embedding.to(x.dtype)
        x = self.encoders.visual.patch_dropout(x)
        x = self.encoders.visual.ln_pre(x)

        features = []

        def hook_fn(module, input, output):
            features.append(output)

        # Hook 등록 부분
        hooks = []
        for n_l in self.num_layers:
            hook = self.encoders.visual.transformer.resblocks[n_l].register_forward_hook(hook_fn)
            hooks.append(hook)

        x = self.encoders.visual.transformer(x)

        # Hook 해제 부분
        for hook in hooks:
            hook.remove()

        return torch.cat([feat.unsqueeze(0) for feat in features]).mean(0)[:, 1:, :]
            
    @torch.no_grad()
    def embed(self, img:torch.Tensor) -> torch.Tensor:        
        image_features = self._embed_img(img)
        return image_features 
    
    @torch.no_grad()
    def fit(self, img_features:np.ndarray):
        if isinstance(img_features,torch.Tensor):
            img_features = img_features.detach().cpu().numpy()
            
        D = img_features.shape[-1]
        #sampling preprocess 
        img_features = img_features.reshape(-1,D).astype(np.float32) # embed dimmension 우선 임시 
        #sampling                      
        sample_features, sample_indices = self.featuresampler.run(img_features)
        # self.anomaly_scorer.fit([sample_features])
        # print('Memory bank fit')
        return sample_features

    @torch.no_grad()
    def predict(self, img:torch.Tensor):
        shape = img.shape
        img_features = self.embed(img)
        img_features = np.asarray(img_features.detach().cpu()).reshape(-1,img_features.shape[-1]).astype(np.float32)        
        
        score, _, indicies = self.anomaly_scorer.predict([img_features])
        
        #image level scoring 
        img_score = self.patchmaker.unpatch_scores(score,shape[0])
        img_score = self.patchmaker.score(img_score)
        
        #Pixel level scoring 
        pixel_score = self.patchmaker.unpatch_scores(score,shape[0])
        pixel_score = pixel_score.reshape(shape[0],int(pixel_score.shape[1]**0.5),-1)
        pixel_score = np.array(self.anomaly_segmentor.convert_to_segmentation(pixel_score))
                
        return score, img_score, pixel_score