# Deep-First Foul Recognition and Contact Boxes

The pipeline is now deep-first:

1. A VARS-style multi-view video model predicts the foul/offence/severity.
2. Red contact boxes are rendered only for actions the model predicts as fouls.
3. The contact-box video freezes for one second when a new contact cue appears.

The contact boxes are an explanation layer. The foul decision comes from the
deep model; the boxes show likely contact regions for review.

## Deep Model

The new deep path is selected with `--pre_model tadaformer_l14`.

It follows the requested TAdaFormer-L/14-style setup:

- 16 frames sampled per view.
- Temporal stride of 2, giving a 32-frame context window.
- Input size `280x490`.
- Random two views during training.
- All available views during validation/test/inference.
- Max pooling across views before the classification heads.
- Learnable view embedding before max pooling:
  - `clip_0` is treated as the live view.
  - `clip_1` to `clip_3` are treated as replay views.
- Two heads:
  - offence/severity classification,
  - action-class classification.

The implementation uses a `timm` ViT-L/14 frame encoder as the TAdaFormer-L/14
adapter. Install `timm` before using this mode.

## Stage One: Fine-Tune Backbone

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model tadaformer_l14 \
  --pooling_type max \
  --num_views 2 \
  --sample_frames 16 \
  --temporal_stride 2 \
  --input_height 280 \
  --input_width 490 \
  --tada_timm_model vit_large_patch14_clip_224.openai \
  --tada_pretrained \
  --max_epochs 20
```

Training uses random two-view sampling for the `Train` split. Validation and
test use all available views.

## Stage Two: Feature Cache and Heads-Only Training

Extract transformer features ten times with random augmentation:

```bash
python scripts/extract_tada_features.py \
  --dataset data/SoccerNet \
  --weights "VARS model/path/to/stage1_model.pth.tar" \
  --output outputs/tada_features \
  --splits Train Valid \
  --repeats 10 \
  --sample-frames 16 \
  --temporal-stride 2 \
  --input-height 280 \
  --input-width 490 \
  --tada-timm-model vit_large_patch14_clip_224.openai
```

Then train only classification heads on the cached features:

```bash
python scripts/train_tada_heads.py \
  --train-features outputs/tada_features/Train_features.pt \
  --valid-features outputs/tada_features/Valid_features.pt \
  --output outputs/tada_heads/tada_heads.pth
```

There is also an online heads-only option if you want to freeze the backbone
without building a feature cache:

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model tadaformer_l14 \
  --path_to_model_weights path/to/stage1_model.pth.tar \
  --freeze_backbone
```

## Evaluation Predictions

After training, produce prediction JSON with the deep model:

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model tadaformer_l14 \
  --pooling_type max \
  --sample_frames 16 \
  --temporal_stride 2 \
  --input_height 280 \
  --input_width 490 \
  --tada_timm_model vit_large_patch14_clip_224.openai \
  --path_to_model_weights path/to/model.pth.tar \
  --only_evaluation 0
```

This writes a `predicitions_test.json` file.

## Red Contact Boxes from Deep Foul Predictions

Use the model prediction JSON to gate the red-box visualizer:

```bash
python scripts/visualize_foul_contact_boxes.py \
  --dataset data/SoccerNet \
  --predictions "VARS model/predicitions_test.json" \
  --splits Test \
  --output outputs/model_gated_contact_boxes \
  --min-contact-area 450 \
  --field-top-ratio 0.18 \
  --contact-distance-ratio 0.08 \
  --contact-motion-p95 8.0 \
  --contact-pause-seconds 1.0 \
  --contact-box-padding 12
```

Only actions predicted as `Offence` get contact-box videos.

## Foul Body-Part Recognition (CV + Deep Context)

This stage splits responsibilities the way the project intends:

- The deep net is the foul **detector** (offence/severity), running on the GPU.
- Classical CV is the foul **recognizer**: for each action the deep net flags
  as a foul, it localizes **where on the body** the contact lands
  (leg, arm, back/torso, shoulder, hip, foot, head).

It combines the existing motion/contact detector (`ClipFeatureExtractor`) with a
`torchvision` Keypoint R-CNN pose estimator (COCO-17 keypoints, GPU). The peak
contact moment is located in the live clip, the players are posed, and the
contact point is mapped to the nearest body region. A coarse Upper/Under-body
label is also produced so the output can be scored against the dataset's
`Bodypart` annotation.

### Where the red box is placed (`--contact-source`)

The red box marks **where contact occurs**. Two strategies are available:

- `pose` (default): the box is placed at the point where two players'
  skeletons are closest. The script poses a window of frames around the centre
  of the clip (where the foul happens), finds the frame and player pair with
  the smallest distance between any two visible keypoints, and draws a tight box
  centred on that contact point. This puts the box on the actual body-to-body
  contact rather than on the strongest motion blob.
- `motion`: the legacy MOG2/optical-flow detector picks the peak-motion contact
  frame and the union box of the interacting blobs. Used as an automatic
  fallback when no two players can be posed.
- `deep`: uses the deep net itself. It occludes a grid of regions in the live
  view and measures how much each one lowers the net's "foulness" score; the
  region whose removal hurts most is where the model is looking. That saliency
  peak is mapped from the model's crop back to the frame and snapped to the
  nearest player, and the heatmap is overlaid. Requires `--weights` (the
  saliency is computed during inline detection). Flags: `--saliency-grid`,
  `--saliency-resize-shorter`, `--saliency-alpha`. Unlike the motion path it can
  localize even when one player is on the ground / poorly posed, because it
  reflects the model's own evidence rather than skeleton proximity.

Pose-source flags:

- `--contact-box-scale` (default `0.4`): box size relative to the involved
  players' height; smaller = tighter box around the contact.
- `--max-pose-frames` (default `24`): frames posed per clip to find the closest
  approach (caps GPU cost).
- `--contact-center-frac` (default `0.6`): restricts the search to the central
  time window of the clip, where the foul occurs.

Keep only genuine two-player contacts:

- `--require-two-players`: skip the action entirely (no motion fallback) when no
  frame poses two players in contact. Skipped actions are reported and excluded
  from the index/outputs.
- `--max-contact-distance-ratio` (default `0` = off): with the flag above,
  reject contacts where the two closest keypoints are farther apart than this
  fraction of the players' height. `0.25` means the two people must nearly
  touch; larger values are more permissive.

### Run with precomputed deep predictions

First produce `predicitions_<split>.json` from the deep net (see "Evaluation
Predictions" above), then:

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Valid \
  --predictions "VARS model/predicitions_valid.json" \
  --output outputs/foul_bodypart \
  --device cuda \
  --render-video \
  --eval
```

### Run the deep net inline (one command, GPU)

Supply the foul-detection checkpoint and let the script run detection itself.
The inline path picks the right preprocessing for the backbone, so pass the
flags that match how the checkpoint was trained.

Original VARS MViT-v2-S checkpoint (e.g. `14_model.pth.tar`, attention pooling,
frames 65-85):

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Valid \
  --weights "VARS model/14_model.pth.tar" \
  --pre-model mvit_v2_s \
  --pooling-type attention \
  --start-frame 65 \
  --end-frame 85 \
  --fps 21 \
  --output outputs/foul_bodypart \
  --device cuda \
  --eval
```

TAdaFormer-L/14 checkpoint:

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Valid \
  --weights "VARS model/path/to/model.pth.tar" \
  --pre-model tadaformer_l14 \
  --pooling-type max \
  --sample-frames 16 \
  --temporal-stride 2 \
  --input-height 280 \
  --input-width 490 \
  --tada-timm-model vit_large_patch14_clip_224.openai \
  --output outputs/foul_bodypart \
  --device cuda \
  --eval
```

Add `--detect-limit N` to cap how many actions the inline deep net scores
(useful for a quick GPU smoke test before a full run).

### Outputs

Per predicted-foul action under `--output/<split>/action_<id>/`:

- `contact_bodypart.png`: contact frame with player skeletons, the highlighted
  contact player, the red contact box/point at the point of contact, the CV
  body-part label, and the deep-net context (offence/severity/action).
- `bodypart.json`: structured result (fine + coarse body part, distance,
  confidence, deep prediction, contact source, contact frame index, contact box).
- `clip_0_overlay.mp4`: full motion red-box overlay (only with `--render-video`).

Plus `index.json` (all records) and, with `--eval`, `bodypart_eval.json`
(coarse accuracy and confusion vs `Bodypart`).

### Body-Part Taxonomy

- Fine (CV output): head, shoulder, arm, back/torso, hip, leg, foot.
- Coarse (scored vs `Bodypart`): head/shoulder/arm/back-torso -> `Upper body`;
  hip/leg/foot -> `Under body`.

### Recognition Tuning

- `--person-score-threshold`: minimum person-detection confidence for pose.
- `--keypoint-score-threshold`: minimum per-keypoint score to use a joint.
- `--peak-strategy`: `contact` (strongest motion spike among possible-contact
  frames) or `closest` (closest two-player approach).
- The contact-box tuning flags below also apply (they decide which frame and
  region the body-part assignment runs on).

The recognizer defaults are the values found by `scripts/tune_bodypart.py`.

### Automated Threshold Tuning

`scripts/tune_bodypart.py` tunes the contact/pose thresholds against the
dataset's ground-truth `Bodypart` label. It caches the expensive per-frame
analysis (pose, and for motion mode optical flow + tracking) once, then sweeps
configs with coordinate ascent in seconds. Two modes:

Pose mode (default, tunes the two-player contact box):

```bash
python scripts/tune_bodypart.py \
  --mode pose \
  --dataset data/SoccerNet \
  --splits Valid \
  --device cuda \
  --min-coverage 0.4
```

It sweeps `--person-score-threshold`, `--keypoint-score-threshold`,
`--contact-center-frac`, `--max-contact-distance-ratio`, and `--max-pose-frames`,
optimizing accuracy on the kept two-player videos subject to a coverage floor
(`--min-coverage`). It writes `outputs/tune/bodypart_tune_report_pose.json` and
prints the recommended recognizer flags. The cache (built once, ~25 min on the
full Valid split because pose runs on every sampled frame) is reused unless you
pass `--rebuild`.

Motion mode (legacy MOG2 box):

```bash
python scripts/tune_bodypart.py --mode motion --dataset data/SoccerNet --splits Valid --device cuda
```

Tuning result on the full Valid split (406 foul actions, 147 Upper / 259 Under):

- Pose default thresholds: 0.473 kept accuracy, 0.860 coverage.
- Pose tuned: 0.483 kept accuracy, 0.948 coverage (the values now used as
  defaults: `person_score=0.4`, `kp_score=2.0`, `center_frac=0.6`,
  `max_distance_ratio=0.35`, `max_pose_frames=32`).
- Legacy motion (`contact` strategy): 0.532.

Caveat: the dataset is 63.8 percent `Under body`, so a majority-class guess
scores ~0.638. Both CV localizers (pose 0.483, legacy motion 0.532) sit below
that majority baseline: they add spatial/explanatory value (they show *where*
on the body and on which players contact occurs) but are weak body-part
*classifiers* on their own. The deep body-part head below is the recommended
classifier; the pose contact box remains the fine-grained visual localizer
(the red box on the point of contact).

## Deep Body-Part Head (recommended classifier)

This learns the coarse Upper/Under body label directly from the frozen deep
net's pooled feature ("context"), while the deep net keeps its foul-detection
role. It clearly beats both the majority baseline and the classical CV method.

### Step 1: cache frozen deep features + Bodypart labels (GPU)

```bash
python scripts/extract_deep_features.py \
  --dataset data/SoccerNet \
  --weights "VARS model/14_model.pth.tar" \
  --splits Train Valid \
  --pre-model mvit_v2_s \
  --pooling-type attention \
  --start-frame 65 --end-frame 85 --fps 21 \
  --device cuda \
  --output outputs/deep_features
```

Short/corrupt clips are skipped automatically. Writes
`Train_bodypart_features.pt` and `Valid_bodypart_features.pt`.

### Step 2: train the head

```bash
python scripts/train_bodypart_head.py \
  --train-features outputs/deep_features/Train_bodypart_features.pt \
  --valid-features outputs/deep_features/Valid_bodypart_features.pt \
  --device cuda \
  --output outputs/bodypart_head/bodypart_head.pth
```

Result on Valid (2295 train / 319 valid foul actions):

| Method | Accuracy | Balanced accuracy |
|---|---|---|
| Majority class (`Under body`) | 0.643 | 0.500 |
| Classical CV (tuned) | 0.532 | - |
| Deep body-part head | 0.740 | 0.739 |

Per-class recall is balanced (Upper 0.74, Under 0.74), so the gain is real and
not just predicting the majority class.

### Step 3: use the head in the recognizer

Pass `--bodypart-head` together with `--weights`. The recognizer then reports
the deep head's coarse Upper/Under prediction (the reliable classifier)
alongside the CV pose overlay (the fine visual localizer):

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Valid \
  --weights "VARS model/14_model.pth.tar" \
  --pre-model mvit_v2_s --pooling-type attention \
  --start-frame 65 --end-frame 85 --fps 21 \
  --bodypart-head outputs/bodypart_head/bodypart_head.pth \
  --device cuda \
  --eval
```

`bodypart.json` then contains both `bodypart` (CV pose fine label) and
`bodypart_deep` (deep head coarse label), and `--eval` reports both accuracies.

## Contact Box Tuning

- `--min-contact-area`: ignores tiny moving fragments.
- `--field-top-ratio`: ignores contact candidates near broadcast graphics.
- `--contact-distance-ratio`: controls how close moving regions must be.
- `--contact-motion-p95`: requires a motion spike for possible contact.
- `--contact-pause-seconds`: controls review pause duration.
- `--contact-box-padding`: controls red-box padding.

## Data

```bash
export SOCCERNET_PASSWORD=""
python scripts/download_mvfoul.py --output data/SoccerNet --version 720p
```

Unzip the downloaded folders so the dataset root contains `Train`, `Valid`,
`Test`, and `Chall`.
