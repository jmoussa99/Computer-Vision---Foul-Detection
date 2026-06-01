# Deep-First VARS + Contact Visualization Pipeline

This project is now focused on one clear goal:

1. A VARS-style deep multi-view video model detects whether an action is a foul.
2. The CV pipeline visualizes likely contact for actions predicted as fouls.
3. Optional pose/body-part analysis explains where the contact appears to land.

The CV code is an explanation and visualization layer. It does not replace the
deep foul detector.

## Deep Foul Detector

Use the original VARS backbones:

- `mvit_v2_s`
- `r2plus1d_18`
- `r3d_18`
- `mc3_18`
- `s3d`

Recommended baseline command for the pretrained VARS MViT-v2-S setup:

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model mvit_v2_s \
  --pooling_type attention \
  --start_frame 65 \
  --end_frame 85 \
  --fps 21 \
  --path_to_model_weights 14_model.pth.tar \
  --only_evaluation 0
```

This writes `predicitions_test.json`. That prediction file is what gates the
contact visualization.

To train the same backbone instead of only evaluating:

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model mvit_v2_s \
  --pooling_type attention \
  --start_frame 65 \
  --end_frame 85 \
  --fps 21 \
  --max_epochs 60
```

## Contact Boxes

Render red contact-box overlays only for actions predicted as fouls:

```bash
python scripts/visualize_foul_contact_boxes.py \
  --dataset data/SoccerNet \
  --predictions "VARS model/predicitions_test.json" \
  --splits Test \
  --output outputs/model_gated_contact_boxes \
  --contact-pause-seconds 1.0
```

The visualizer:

- skips actions predicted as `No offence`;
- draws red boxes around likely contact regions;
- pauses the output video for one second when contact is detected;
- writes per-action `features.json` plus an output `index.json`.

## Body-Part Contact Recognition

For a stronger contact visualization, use pose estimation to place the red box
where two player skeletons are closest:

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

You can also run the deep detector inline from a VARS checkpoint:

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

Useful options:

- `--contact-source pose`: default; red box comes from closest skeleton contact.
- `--contact-source motion`: red box comes from motion/contact blobs.
- `--contact-source deep`: red box comes from occlusion saliency over the deep
  model's foul score.
- `--require-two-players`: skip actions when pose cannot find a clear two-player
  contact.
- `--max-contact-distance-ratio 0.25`: require the two posed players to be very
  close before accepting a contact.

Outputs under `--output/<split>/action_<id>/`:

- `contact_bodypart.png`: contact frame with skeletons and red contact box.
- `bodypart.json`: prediction, contact box, body-part assignment, and metadata.
- `clip_0_overlay.mp4`: optional full overlay video when `--render-video` is set.
- `index.json`: project-level index of processed actions.
- `bodypart_eval.json`: coarse body-part accuracy when `--eval` is set.

## CV Feature Fusion

The classical CV feature extractor can still create compact motion/contact
features for late fusion with the VARS model:

```bash
python scripts/extract_cv_features.py \
  --dataset data/SoccerNet \
  --splits Train Valid Test \
  --output outputs/cv_features
```

Then train/evaluate VARS with those features:

```bash
cd "VARS model"
python main.py \
  --path ../data/SoccerNet \
  --pre_model mvit_v2_s \
  --pooling_type attention \
  --cv_features_path ../outputs/cv_features \
  --cv_feature_set core
```

## Pipeline Summary

The deep VARS baseline answers: **is this action a foul, and what type/severity
is it?**

The CV pipeline answers: **where does the visible contact seem to happen, and
how can we show it on the video?**
