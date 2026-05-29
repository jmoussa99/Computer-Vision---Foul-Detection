
import __future__
import torch
from mvaggregate import MVAggregate
from tadaformer import TadaFormerL14MVNetwork
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights



class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18', agr_type='max', lifting_net=torch.nn.Sequential(), cv_feat_dim=0, tada_pretrained=True, tada_input_size=(280, 490), tada_timm_model="vit_large_patch14_clip_224.openai"):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        self.cv_feat_dim = cv_feat_dim
        self.is_tadaformer = net_name == "tadaformer_l14"

        if self.is_tadaformer:
            self.mvnetwork = TadaFormerL14MVNetwork(
                pretrained=tada_pretrained,
                input_size=tada_input_size,
                cv_feat_dim=cv_feat_dim,
                timm_model=tada_timm_model,
            )
            return
        
        self.feat_dim = 512

        if net_name == "r3d_18":
            weights_model = R3D_18_Weights.DEFAULT
            network = r3d_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "r2plus1d_18":
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
        elif net_name == "mvit_v2_s":
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        else:
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
                
        network.fc = torch.nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
            lifting_net=self.lifting_net,
            cv_feat_dim=self.cv_feat_dim,
        )

    def forward(self, mvimages, cv_features=None, view_ids=None):
        if self.is_tadaformer:
            return self.mvnetwork(mvimages, cv_features=cv_features, view_ids=view_ids)
        return self.mvnetwork(mvimages, cv_features=cv_features)

    def extract_features(self, mvimages, cv_features=None, view_ids=None):
        if not self.is_tadaformer:
            raise NotImplementedError("Feature extraction is currently implemented for tadaformer_l14.")
        return self.mvnetwork.extract_features(mvimages, cv_features=cv_features, view_ids=view_ids)

    def freeze_backbone(self):
        if self.is_tadaformer:
            self.mvnetwork.freeze_backbone()
