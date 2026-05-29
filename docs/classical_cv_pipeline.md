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
