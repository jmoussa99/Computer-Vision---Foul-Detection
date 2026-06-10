# Football Foul Detection Setup

This project detects football fouls with a deep multi-view video model and then
renders visual contact evidence with red boxes, pose overlays, body-part
estimates, videos, and JSON summaries.

## 1. Create the Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your machine first. For a GPU machine, install the CUDA
version that matches your driver from the PyTorch install page.

Then install the project dependencies:

```bash
pip install -r requirements.txt
pip install SoccerNet pyav
```

## 2. Download the Dataset

Set the SoccerNet password as an environment variable:

```bash
export SOCCERNET_PASSWORD="<your SoccerNet password>"
```

Download SoccerNet-MVFoul:

```bash
python scripts/download_mvfoul.py \
  --output data/SoccerNet \
  --version 720p
```

After extraction, the dataset folder should look like:

```text
data/SoccerNet/
  Train/
  Valid/
  Test/
  Chall/
```

## 3. Add Model Weights

Place the VARS checkpoint in:

```text
VARS model/14_model.pth.tar
```

The main scripts below assume that path.

## 4. Run the Deep Foul Detector

From the repository root:

```bash
cd "VARS model"

python main.py \
  --path ../data/SoccerNet \
  --pre_model mvit_v2_s \
  --pooling_type attention \
  --start_frame 65 \
  --end_frame 85 \
  --fps 21 \
  --path_to_model_weights 14_model.pth.tar

cd ..
```

This produces prediction JSON files such as:

```text
VARS model/predicitions_test.json
VARS model/predicitions_valid.json
```

## 5. Render Simple Red Contact Boxes

Use the prediction JSON to render red boxes only for actions predicted as fouls:

```bash
python scripts/visualize_foul_contact_boxes.py \
  --dataset data/SoccerNet \
  --predictions "VARS model/predicitions_test.json" \
  --splits Test \
  --output outputs/model_gated_contact_boxes \
  --contact-pause-seconds 1.0
```

## 6. Run Pose and Body-Part Contact Visualization

Run the full contact visualization pipeline:

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Test \
  --weights "VARS model/14_model.pth.tar" \
  --pre-model mvit_v2_s \
  --pooling-type attention \
  --start-frame 65 \
  --end-frame 85 \
  --fps 21 \
  --output outputs/foul_bodypart_test_all_videos \
  --device cuda \
  --contact-source hybrid \
  --clip-selection replay-closeup \
  --include-original-clip \
  --all-selected-clips \
  --foul-box-window 3 \
  --eval
```

Main outputs:

```text
outputs/foul_bodypart_test_all_videos/index.json
outputs/foul_bodypart_test_all_videos/bodypart_eval.json
outputs/foul_bodypart_test_all_videos/<split>/action_<id>/clip_<n>_foul_box.mp4
outputs/foul_bodypart_test_all_videos/<split>/action_<id>/clip_<n>_foul_detection.mp4
outputs/foul_bodypart_test_all_videos/<split>/action_<id>/clip_<n>_contact_bodypart.png
outputs/foul_bodypart_test_all_videos/<split>/action_<id>/clip_<n>_bodypart.json
```

## 7. Optional: Train the Body-Part Head

Extract frozen deep features:

```bash
python scripts/extract_deep_features.py \
  --dataset data/SoccerNet \
  --splits Train Valid \
  --weights "VARS model/14_model.pth.tar" \
  --pre-model mvit_v2_s \
  --output outputs/deep_features
```

Train the small Upper/Under body-part classifier:

```bash
python scripts/train_bodypart_head.py \
  --train-features outputs/deep_features/Train_bodypart_features.pt \
  --valid-features outputs/deep_features/Valid_bodypart_features.pt \
  --output outputs/bodypart_head/bodypart_head.pth \
  --report outputs/bodypart_head/bodypart_head_report.json
```

Use it during contact visualization:

```bash
python scripts/recognize_foul_bodypart.py \
  --dataset data/SoccerNet \
  --splits Test \
  --weights "VARS model/14_model.pth.tar" \
  --bodypart-head outputs/bodypart_head/bodypart_head.pth \
  --output outputs/foul_bodypart_with_head \
  --device cuda \
  --contact-source hybrid \
  --clip-selection replay-closeup \
  --include-original-clip \
  --all-selected-clips \
  --foul-box-window 3 \
  --eval
```

## 8. Useful Existing Result Folders

```text
outputs/foul_bodypart_replays_plus_original_windowed/
outputs/foul_bodypart_replays_plus_original_windowed_chall/
outputs/foul_bodypart_test_all_videos/
outputs/bodypart_head/
```

## 9. Project Documents

```text
docs/classical_cv_pipeline.md
docs/foul_detection_final_results_presentation.pptx
docs/computer_vision_foul_detection_paper.docx
computer_vision_foul_detection_paper.pdf
```
