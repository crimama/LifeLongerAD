import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm import create_model
import cv2 
from.sampler import ApproximateGreedyCoresetSampler
from .common import PatchMaker, NearestNeighbourScorer, RescaleSegmentor, FaissNN

class UCAD(nn.Module):
    def __init__(self,class_names:list,backbone:str='vit_base_patch16_224',input_shape:list=[3,224,224], num_blocks:int=5,prompt_dim:int=768,
                 key_embedding='mean', n_neighbors:int=5,faiss_on_gpu:bool=False,faiss_num_workers:int=4):
        super(UCAD,self).__init__()
        self.vit = ViT(backbone=backbone,num_blocks=num_blocks)
        self.cpm = ContinualPromptingModule(class_names=class_names,prompt_dim=prompt_dim,num_blocks=num_blocks,key_embedding=key_embedding)
        self.scl = StructureBasedContrastiveLearning()
        
        self.patchmaker = PatchMaker(patchsize=3,stride=1)
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=n_neighbors, nn_method=FaissNN(on_gpu = faiss_on_gpu, num_workers = faiss_num_workers)
            )
        self.anomaly_segmentor = RescaleSegmentor(
            target_size=input_shape[-2:]
            )        
        
        
    def forward(self, img, prompts=None) -> torch.Tensor:
        x = self.vit(img, prompts)
        return x 
    
    def criterion(self, feats:torch.Tensor, img_dirs:list):
        return self.scl(feats,img_dirs)
    
    # def predict(self, images:torch.Tensor, class_name:str):        
    #     knowledge = self.cpm.knowledge[class_name]
    #     prompts = self.cpm.prompts[class_name]
        
    #     features = self.vit(images, prompts)[-1]
    #     # scores = torch.norm((features - knowledge.unsqueeze(0).to(features.device)),dim=-1) # Calculate score btw knowledge-query          #! 이 방식으로 했을 때 더 image level 성능 잘나옴 
    #     # image_level_score = scores.max(1)[0]
        
    #     distances = torch.norm(features.unsqueeze(2) - 
    #                             knowledge.unsqueeze(0).unsqueeze(0).to(features.device), dim=-1) # [B, N_features, N_knowledge]
    #     #distances = torch.topk(distances, k=5, largest=False, dim=-1).values.max(dim=-1).values
    #     image_level_score = distances.max(-1).values.max(-1).values        
        
    #     B = distances.shape[0] #! 여기 수정 필요 
    #     P = 196 #! 
    #     pixel_level_score  = F.interpolate(
    #                 distances.view(B,int(P**0.5),-1).unsqueeze(0), size=images.shape[-1], mode="bilinear", align_corners=False
    #             ).squeeze(0)
    #     return image_level_score, pixel_level_score 
    
    @torch.no_grad()
    def predict(self, img:torch.Tensor, class_name:str=None):
        prompts = self.cpm.prompts[class_name]
        knowledge = self.cpm.knowledge[class_name]        
        self.anomaly_scorer.fit([knowledge])
        
        shape = img.shape
        # img_features = self.vit(img, prompts)[-1]
        img_features = self.vit(img)
        img_features = torch.cat([f.unsqueeze(1) for f in img_features],dim=1).mean(dim=1)
        img_features = np.asarray(img_features.detach().cpu()).reshape(-1,768).astype(np.float32)        
        score, _, indicies = self.anomaly_scorer.predict([img_features])
        
        #image level scoring 
        img_score = self.patchmaker.unpatch_scores(score,shape[0])
        img_score = self.patchmaker.score(img_score)
        
        #Pixel level scoring 
        pixel_score = self.patchmaker.unpatch_scores(score,shape[0])
        pixel_score = pixel_score.reshape(shape[0],int(pixel_score.shape[1]**0.5),-1)
        pixel_score = np.array(self.anomaly_segmentor.convert_to_segmentation(pixel_score))
                
        return score, img_score, pixel_score
    
    
        

class ViT(nn.Module):
    def __init__(self, backbone='vit_base_patch16_224', num_blocks: int = 5):
        super(ViT, self).__init__()
        self.backbone = create_model(backbone, pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        # Extract features and add prompts
        features = self.backbone.patch_embed(x)
        
        cls_token = self.backbone.cls_token.expand(features.shape[0], -1, -1)
        x = torch.cat((cls_token, features), dim=1)
        
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # hook_outputs = []  # hook outputs 초기화
        # for i, blk in enumerate(self.backbone.blocks):
        #     if prompts is not None and i < len(prompts):
        #         if i ==0:
        #             prompt = prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
        #             x = torch.cat([x, prompt], dim=1)
        #         else:
        #             x[:,-1,:] = prompts[i]
                    
        hook_outputs = []  # hook outputs 초기화
        for i, blk in enumerate(self.backbone.blocks):
            if prompts is not None and i < len(prompts):
                prompt = prompts[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                # 매번 프롬프트를 torch.cat으로 결합하여 덮어씁니다.
                
                if i ==0:
                    x = torch.cat([x,prompt],dim=1)
                else:
                    x = torch.cat([x[:, :-1, :], 
                                    prompt
                                    ], dim=1)
            x = blk(x)
            hook_outputs.append(x) 
            if i == (self.num_blocks-1):
                break 
        
        if prompts is not None:
            return [hook[:, 1:-1, :] for hook in hook_outputs]
        else:
            return [hook[:, 1:, :] for hook in hook_outputs]
        
class Prompts(nn.Module):
    def __init__(self, num_blocks: int = 5, prompt_dim: int = 768):
        super(Prompts, self).__init__()
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.nn.init.uniform_(torch.randn(prompt_dim), -1, 1), requires_grad=True)
            for _ in range(num_blocks)
        ])
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]

class ContinualPromptingModule(nn.Module):
    def __init__(self, class_names:list, num_blocks:int=5, prompt_dim=768, 
                key_embedding='mean', feature_sampler = ApproximateGreedyCoresetSampler, num_samples = 196):
        '''
            class_names : Incremental class names (ordered)
            num_blocks : last number of block of ViT, which we use for knowledge 
            key_embedding : Method to make Key pool : mean, FPS, max, 
            num_samples : The number of sampled feature for knowledge 
        '''
        super(ContinualPromptingModule, self).__init__()
        # self.prompts = nn.ModuleDict({
        #     cn: nn.ParameterList([nn.Parameter(torch.randn(prompt_dim)) for _ in range(num_blocks)])
        #     for cn in class_names
        # })
        self.prompts = {cn: Prompts(num_blocks=num_blocks,prompt_dim=prompt_dim) for cn in class_names}
        self.keys = {}
        self.knowledge = {}
        
        self.num_blocks = num_blocks
        self.key_embedding = key_embedding
        self.featuresampler = ApproximateGreedyCoresetSampler(num_samples=num_samples*2) #? knowledge pool size control하게 추후 수정 필요 
        self.class_names = list(class_names)
        self.key_i_dict = {cn:i for i,cn in enumerate(class_names)}
        
    #! Key 
    def get_key(self, key_pool:torch.Tensor, class_name:str):
        '''
            Input : Key_pool : [B,196,D]
            Output : Key_pool : [196,D]
        '''
        B,P,D = key_pool.shape
        if self.key_embedding == 'mean':
            key_pool = torch.mean(key_pool,dim=0)            
        elif self.key_embedding == 'fps':
            key_pool = self.furthest_point_sampling_embeddings(key_pool.reshape(-1,D),P).detach().cpu()
        elif self.key_embedding == 'max':
            key_pool = torch.max(key_pool,dim=0)
        self.keys[class_name] = key_pool 
        print(f'{class_name} key save done')        
        
    def furthest_point_sampling_embeddings(self, embeddings, num_samples):
        # 모든 텐서를 GPU로 이동시킵니다.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = embeddings.to(device)

        n_points = embeddings.shape[0]
        centroids = [torch.randint(0, n_points, (1,), device=device).item()]

        # 거리를 inf로 초기화합니다.
        distances = torch.full((n_points,), float('inf'), device=device)

        # num_samples-1번 반복합니다.
        for _ in range(num_samples - 1):
            # 마지막 선택된 중심과 모든 포인트 간의 거리 계산
            centroid = embeddings[centroids[-1]].unsqueeze(0)  # (1, D)
            current_distances = torch.cdist(embeddings, centroid).squeeze(1)  # (N,)
            
            # 기존 거리와 새로운 거리 중 작은 값으로 업데이트
            distances = torch.min(distances, current_distances)
            
            # 가장 먼 지점을 다음 중심으로 선택
            centroids.append(torch.argmax(distances).item())

        return embeddings[centroids]

    #! Retrieve Key 
    def retrieve_key(self, query_features:torch.Tensor) -> list:
        '''
            Input : query features 
            Output : Query result, which mean what class 
        '''
        sampled_features = torch.cat([k.unsqueeze(0) for k in list(self.keys.values())]).to(query_features.device)
        sampled_features_flat = sampled_features.view(sampled_features.shape[0], -1)  # [num_samples, 196 * 768]
        new_features_flat = query_features.view(query_features.shape[0], -1)  # [m, 196 * 768]
        
        distances = torch.cdist(new_features_flat.unsqueeze(1), sampled_features_flat.unsqueeze(0), p=2).squeeze(1)  # [m, num_samples]
        min_dist, min_indices = torch.min(distances, dim=1)  # [m]
        query_result = [list(self.keys.keys())[i.item()] for i in min_indices]
        return [self.key_i_dict[k] for k in query_result]
        
    #! Prompts 
    def get_prompts(self, class_name:str, device:str):
        return self.prompts[class_name].to(device)
    
    def detach_prompts(self,class_name:str,prompts:list):
        self.prompts[class_name] = [p.detach().cpu() for p in prompts]
        
    #! knowledge 

    def retrieve_knowledge(self, class_id):
        # Retrieve stored knowledge for a given class
        return self.key_prompt_knowledge.get(class_id), self.coreset_knowledge.get(class_id)
    
    def get_knowledge(self, knowledge_pool:torch.Tensor, class_name:str)-> None:
        '''
            Input :
                - Knowledge_pool : [B,P,D]
                - class_name 
        '''
        B,P,D = knowledge_pool.shape
        sampled_features = self.featuresampler.run(knowledge_pool.reshape(-1,D))
        self.knowledge[class_name] = sampled_features.detach().cpu()
        print(f'{class_name} Knowledge save done')
    
class StructureBasedContrastiveLearning(nn.Module):
    def __init__(self):
        super(StructureBasedContrastiveLearning, self).__init__()

    def forward(self, features: torch.Tensor, img_dirs: list) -> torch.float:
        """
        Input
            - features : visual embeddings of input images from n block
            - img_dirs : list of image directories of input images to get SAM mask
        Output
            - loss : contrastive loss
        """
        # Extract labels efficiently using a list comprehension
        labels = torch.stack([self.get_labels(img_dir) for img_dir in img_dirs]).cuda()
        
        # Compute contrastive loss
        loss = self.contrastive_loss(features, labels.to(features.device), temperature=0.5)
        return loss

    def get_labels(self, img_dir: str) -> torch.Tensor:
        """
        Extracts and resizes SAM mask based on the image directory.
        """
        if 'MVTecAD' in img_dir:
            sam_score = cv2.imread(img_dir.replace('MVTecAD', 'mvtec2d-sam-b'), cv2.IMREAD_GRAYSCALE)
        elif 'visa' in img_dir:
            sam_score = cv2.imread(img_dir.replace('visa', 'visa-sam-b'), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Unknown dataset in image directory")
        
        resized_score = cv2.resize(sam_score, (14, 14)).flatten()
        return torch.from_numpy(resized_score).float()

    # def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor, temperature=0.5) -> torch.Tensor:
    #     # Normalize features
    #     features_normalized = F.normalize(features, dim=2)

    #     # Calculate similarity matrix
    #     similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2)) / temperature

    #     # Create mask
    #     mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

    #     # Positive pairs (maximize similarity)
    #     positive_loss = -torch.log(torch.exp(similarity_matrix) * mask + 1e-8).sum() / mask.sum()

    #     # Negative pairs (minimize similarity)
    #     negative_loss = torch.log(1 + torch.exp(similarity_matrix) * (1 - mask)).sum() / (1 - mask).sum()

    #     # Total loss
    #     loss = positive_loss + negative_loss

    #     return loss
    def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor, temperature=0.5) -> torch.Tensor:
        # Normalize features
        features_normalized = F.normalize(features, dim=2)

        # Calculate similarity matrix
        similarity_matrix = torch.bmm(features_normalized, features_normalized.transpose(1, 2)) / temperature

        # Create mask
        mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

        # Positive pairs (maximize similarity)
        positive_loss = -torch.log(torch.clamp(torch.exp(similarity_matrix) * mask, min=1e-8)).sum() / mask.sum()

        # Negative pairs (minimize similarity)
        negative_loss = torch.log1p(torch.exp(similarity_matrix) * (1 - mask)).sum() / (1 - mask).sum()

        # Total loss
        loss = positive_loss + negative_loss

        return loss

