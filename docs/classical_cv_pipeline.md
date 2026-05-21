# Visual CV Pipeline for SoccerNet-MVFoul

This project keeps the upstream VARS multi-view deep model and adds a focused
computer-vision pipeline for the project goal: visually tracking player
movement and highlighting likely contact moments.

The active CV pipeline uses only:

- 2D motion analysis with dense optical flow.
- Motion-region tracking with foreground masks and centroid association.
- Contact-proxy cues from close tracked regions and motion spikes.
- Annotated overlay videos for visual inspection.

## Download Data

```bash
export SOCCERNET_PASSWORD=''
python scripts/download_mvfoul.py --output data/SoccerNet --version 720p
```

Unzip the downloaded folders so the dataset root contains `Train`, `Valid`,
`Test`, and `Chall`, matching the upstream README.

## Create Visual Tracking Overlays

Quick run on the bundled interface sample:

```bash
python scripts/extract_cv_features.py \
  --dataset "VARS interface/dataset" \
  --splits . \
  --output outputs/contact_demo \
  --max-actions 1 \
  --visualize \
  --contact-distance-ratio 0.10 \
  --contact-motion-p95 6.0
```

Run on the full MVFoul data:

```bash
python scripts/extract_cv_features.py \
  --dataset data/SoccerNet \
  --output outputs/cv_features \
  --splits Train Valid Test Chall \
  --visualize
```

Each action gets a `features.json` and, with `--visualize`, one overlay video
per clip. `outputs/cv_features/index.json` lists all processed actions.

The overlay videos draw:

- green boxes and IDs around moving player/object regions,
- white motion trails for each tracked region,
- orange lines when two tracked regions are close,
- red lines and a `POSSIBLE CONTACT` banner when closeness coincides with a
  strong optical-flow motion spike.

## Contact Sensitivity

Use these two knobs to make the visualization stricter or looser:

- `--contact-distance-ratio`: normalized distance threshold for close tracked
  regions. Larger values create more close/contact candidates.
- `--contact-motion-p95`: optical-flow motion-spike threshold. Lower values
  label more close interactions as possible contact.

Example stricter contact display:

```bash
python scripts/extract_cv_features.py \
  --dataset "VARS interface/dataset" \
  --splits . \
  --output outputs/contact_strict \
  --max-actions 1 \
  --visualize \
  --contact-distance-ratio 0.07 \
  --contact-motion-p95 10.0
```

## Foul Detection Model Fusion

The same visual tracking/contact descriptors can be fused into the upstream
VARS model with late fusion. The video backbone still encodes the multi-view
clips, while a small MLP encodes the `core` CV vector and concatenates it before
the action and offence/severity heads.

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --cv_features_path ../outputs/cv_features \
  --cv_feature_set core \
  --pooling_type attention \
  --pre_model mvit_v2_s
```

If you load original VARS weights while fusion is enabled, compatible backbone
weights are reused and the new CV-fusion layers are initialized from scratch.

## Classical Baseline

For a lightweight non-deep baseline over the visual CV descriptors:

```bash
python scripts/train_classical_cv.py \
  --features outputs/cv_features \
  --dataset data/SoccerNet \
  --target offence_severity \
  --feature-set core \
  --train-split Train \
  --eval-split Valid \
  --output outputs/classical_cv
```
