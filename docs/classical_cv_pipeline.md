# Classical CV Pipeline for SoccerNet-MVFoul

This project keeps the upstream VARS multi-view deep model and adds a visual
computer-vision pipeline for tracking player movement and highlighting likely
contact moments. The default classifier uses the same core signals:

- 2D motion analysis: dense Farneback optical-flow magnitude statistics.
- Object tracking: motion-mask detections linked with a centroid tracker.
- Contact-proxy interaction cues: close moving-object pairs and motion spikes
  during close interactions.
- Local visual features: ORB keypoints and cross-view feature matches.

Edge detection is included as a lightweight baseline signal in the default
`core` feature set and can be evaluated alone with `--feature-set
edge_baseline`. The following are optional extensions for demos, reports, or
experiments:

- Image stitching: ORB + RANSAC homography with a first-frame panorama.
- Camera calibration and pose estimation: chessboard intrinsic/extrinsic solve.
- Stereo vision: first-frame block-matching disparity between two views.

## Download Data

Do not hard-code the SoccerNet password. Use an environment variable:

```bash
export SOCCERNET_PASSWORD=''
python scripts/download_mvfoul.py --output data/SoccerNet --version 720p
```

Unzip the downloaded folders so the dataset root contains `Train`, `Valid`,
`Test`, and `Chall`, matching the upstream README.

## Extract Features

Quick run on the bundled interface sample:

```bash
python scripts/extract_cv_features.py \
  --dataset "VARS interface/dataset" \
  --splits . \
  --output outputs/interface_cv \
  --max-actions 5 \
  --visualize
```

The `--visualize` overlays are the main visual deliverable. They draw:

- green boxes and IDs around moving player/object regions,
- white motion trails for each tracked region,
- orange lines when two moving regions are close,
- red lines and a `POSSIBLE CONTACT` banner when closeness coincides with a
  strong optical-flow motion spike.

Tune the contact display with:

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

Each action gets a `features.json` and optional overlay videos.
`outputs/cv_features/index.json` lists all outputs.

To run optional stitching and stereo experiments, add `--stitch --stereo` to
the extraction command. Those outputs are useful for visualization and
multi-camera geometry checks, but they are not required for the classifier.

## Classical Foul Classifier

Once Train and Valid features have been extracted, fit a traditional ML model:

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

The trainer uses a balanced RandomForest over the `core` descriptor set: edge
baseline, motion, tracking, contact-proxy interaction cues, and cross-view
matching. It writes a pickle model and a JSON evaluation report. Supported
targets are `action`, `offence`, `severity`, and `offence_severity`.

Useful ablations:

```bash
python scripts/train_classical_cv.py --features outputs/cv_features --dataset data/SoccerNet --feature-set motion_tracking_local
python scripts/train_classical_cv.py --features outputs/cv_features --dataset data/SoccerNet --feature-set edge_baseline
```

## VARS Fusion

The extracted CV descriptors can be fused into the upstream VARS model with
late fusion. The video backbone still encodes the multi-view clips, while a
small MLP encodes the `core` CV vector and concatenates it before the action and
offence/severity heads.

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

## Camera Calibration

Use chessboard photos from the camera you want to calibrate:

```bash
python scripts/calibrate_camera.py \
  --images data/calibration/camera_0 \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output outputs/calibration/camera_0.json
```

The JSON contains the camera matrix, distortion coefficients, rotation vectors,
translation vectors, and reprojection error.

## Train the VARS Baseline

The original training entry point remains:

```bash
cd "VARS model"
python main.py --path ../data/SoccerNet --pooling_type attention --pre_model mvit_v2_s
```

The classical feature pipeline is intentionally separate so you can use it for
visual analysis, reports, or later fusion without disturbing the baseline model.
