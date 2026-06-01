from torch.utils.data import Dataset
import torch
import random
import json
import sys
from pathlib import Path
from data_loader import label2vectormerge, clips2vectormerge
try:
    from torchvision.io import read_video
except (ImportError, AttributeError):
    try:
        from torchvision.io.video import read_video
    except (ImportError, AttributeError):
        read_video = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_foul_detection.classical_model import feature_dim, vectorize_payload


def read_video_frames(path):
    if read_video is not None:
        video, _, _ = read_video(path, output_format="THWC")
        return video

    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))
    capture.release()

    if not frames:
        raise RuntimeError(f"Could not read video: {path}")
    return torch.stack(frames, dim=0)


class MultiViewDataset(Dataset):
    def __init__(
        self,
        path,
        start,
        end,
        fps,
        split,
        num_views,
        transform=None,
        transform_model=None,
        cv_features_path="",
        cv_feature_set="core",
    ):

        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset. 
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0]. 

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset. 
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0]. 

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor
        
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views
        self.cv_features_path = Path(cv_features_path) if cv_features_path else None
        self.cv_feature_set = cv_feature_set
        self.cv_feature_dim = feature_dim(cv_feature_set) if self.cv_features_path else 0
        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action, 
    def getCvFeatureDim(self):
        return self.cv_feature_dim


    # RETURNS
    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):

        prev_views = []
        for num_view in range(len(self.clips[index])):

            index_view = num_view

            if len(prev_views) == 2:
                continue

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can have 2, 3 or 4 views per action.
            cont = True
            if self.split == 'Train':
                while cont:
                    aux = random.randint(0,len(self.clips[index])-1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)

            video = read_video_frames(self.clips[index][index_view])
            frames = video[self.start:self.end,:,:,:]
            final_frames = None
            for j in range(len(frames)):
                if j%self.factor<1:
                    if final_frames == None:
                        final_frames = frames[j,:,:,:].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

            final_frames = final_frames.permute(0, 3, 1, 2)

            if self.transform != None:
                final_frames = self.transform(final_frames)

            if self.transform_model != None:
                final_frames = self.transform_model(final_frames)
            final_frames = final_frames.permute(1, 0, 2, 3)
            
            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        videos = videos.permute(0, 2, 1, 3, 4)

        if self.split != 'Chall':
            action_id = self.number_of_actions[index]
            if self.cv_features_path:
                return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self._load_cv_features(action_id), action_id
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, action_id
        else:
            action_id = str(index)
            if self.cv_features_path:
                return -1, -1, videos, self._load_cv_features(action_id), action_id
            return -1, -1, videos, action_id

    def __len__(self):
        return self.length

    def _load_cv_features(self, action_id):
        feature_file = self.cv_features_path / self.split / ("action_" + str(action_id)) / "features.json"
        if not feature_file.exists():
            return torch.zeros(self.cv_feature_dim, dtype=torch.float32)
        with feature_file.open(encoding="utf-8") as f:
            payload = json.load(f)
        return torch.tensor(vectorize_payload(payload, self.cv_feature_set), dtype=torch.float32)

