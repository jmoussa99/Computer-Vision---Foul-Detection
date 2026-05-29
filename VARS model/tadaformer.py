import torch
from torch import nn
import torch.nn.functional as F


class TadaFormerL14MVNetwork(nn.Module):
    """TAdaFormer-L/14-style multi-view recognizer.

    This adapter follows the project design:
    - 16 frames per view, sampled outside the model with stride 2.
    - a ViT-L/14 frame encoder from timm when available,
    - learnable live/replay view embeddings before multi-view max pooling,
    - separate offence/severity and action classification heads.

    The official TAdaFormer implementation is not part of torchvision, so this
    class keeps the integration point isolated while preserving the expected
    VARS-style input/output contract.
    """

    def __init__(self, pretrained=True, input_size=(280, 490), cv_feat_dim=0, timm_model="vit_large_patch14_clip_224.openai"):
        super().__init__()
        self.input_size = input_size
        self.cv_feat_dim = cv_feat_dim
        self.frame_encoder, self.feat_dim = self._build_frame_encoder(timm_model, pretrained)
        self.view_embedding = nn.Embedding(2, self.feat_dim)

        fused_dim = self.feat_dim
        if self.cv_feat_dim > 0:
            self.cv_encoder = nn.Sequential(
                nn.LayerNorm(self.cv_feat_dim),
                nn.Linear(self.cv_feat_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(),
            )
            fused_dim += self.feat_dim

        self.inter = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
        )
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, 4),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, 8),
        )

    def forward(self, mvclips, cv_features=None, view_ids=None):
        features, view_features = self.extract_features(mvclips, cv_features=cv_features, view_ids=view_ids)
        inter = self.inter(features)
        pred_offence_severity = self.fc_offence(inter)
        pred_action = self.fc_action(inter)
        return pred_offence_severity, pred_action, view_features

    def extract_features(self, mvclips, cv_features=None, view_ids=None):
        batch_size, num_views, channels, frames, height, width = mvclips.shape
        frames_flat = mvclips.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * num_views * frames, channels, height, width)
        if (height, width) != self.input_size:
            frames_flat = F.interpolate(frames_flat, size=self.input_size, mode="bilinear", align_corners=False)

        frame_features = self.frame_encoder(frames_flat)
        view_features = frame_features.reshape(batch_size, num_views, frames, self.feat_dim).mean(dim=2)

        if view_ids is None:
            view_ids = torch.arange(num_views, device=mvclips.device).unsqueeze(0).expand(batch_size, -1)
        view_types = (view_ids > 0).long().to(mvclips.device)
        view_features = view_features + self.view_embedding(view_types)

        pooled = torch.max(view_features, dim=1)[0]
        if self.cv_feat_dim > 0:
            if cv_features is None:
                cv_features = torch.zeros((batch_size, self.cv_feat_dim), device=mvclips.device, dtype=mvclips.dtype)
            if len(cv_features.shape) == 1:
                cv_features = cv_features.unsqueeze(0)
            cv_features = cv_features.to(device=mvclips.device, dtype=mvclips.dtype)
            pooled = torch.cat((pooled, self.cv_encoder(cv_features)), dim=1)
        return pooled, view_features

    def freeze_backbone(self):
        for param in self.frame_encoder.parameters():
            param.requires_grad = False

    def _build_frame_encoder(self, model_name, pretrained):
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("TAdaFormer-L/14 mode requires timm. Install it with `pip install timm`.") from exc

        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=self.input_size,
                dynamic_img_size=True,
            )
        except TypeError:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = getattr(model, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"Could not infer feature dimension for timm model: {model_name}")
        return model, feat_dim
