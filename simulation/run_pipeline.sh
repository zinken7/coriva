#!/bin/bash
set -e

# Resolve script directory (simulation/)
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SIM_DIR")"
RAW_DIR="$WORKSPACE/out"
OUTPUT_DIR="$WORKSPACE/output"
MAIN_JSON="$RAW_DIR/coriva_bench_examples.json"
GT_DIR="$WORKSPACE/data/ground_truth"
FPS=10
INTERVENTION_TIME=3.0
# Number of frames before intervention: 1 init tick + 30 steps (0-29)
PRE_FRAMES=31

echo "========================================="
echo "   CORIVA BENCHMARK - FULL PIPELINE"
echo "========================================="

# ═══════════════════════════════════════════
# PHASE 1: SIMULATION
# ═══════════════════════════════════════════
echo -e "\n[Phase 1] Cleaning up old data..."
rm -rf "$RAW_DIR" "$OUTPUT_DIR"
echo "-> Cleanup complete!"

echo -e "\n[Phase 1] Running CARLA simulation..."
cd "$SIM_DIR"
python3 "$SIM_DIR/generate.py"

echo -e "\n[Phase 1] Removing corrupted frames (0 bytes)..."
find "$RAW_DIR" -type f -size 0 -delete

# ═══════════════════════════════════════════
# PHASE 2: PROCESS (videos, JSONs, overlays)
# ═══════════════════════════════════════════
echo -e "\n[Phase 2] Processing output..."

# Install opencv if not present (needed for validate_overlay.py)
pip install -q opencv-python-headless 2>/dev/null || true

# --- Function to process an intervention branch ---
process_intervention() {
    local raw_sc_dir="$1"       # e.g. /workspace/out/scenario_1
    local branch="$2"           # e.g. Factual_AEB
    local output_dir="$3"       # e.g. .../output/scenario_1/interventions/Factual_AEB
    local json_scenario="$4"    # e.g. high_speed_following
    local target_key="$5"       # e.g. npc

    mkdir -p "$output_dir"

    local frames_dir="$raw_sc_dir/$branch"
    local vid_name="${branch}.mp4"
    local json_name="${branch}.json"
    local validated_name="validated_${branch}.mp4"

    # 1. FFmpeg: join all frames → video
    ffmpeg -loglevel warning -y -framerate $FPS -pattern_type glob \
        -i "$frames_dir/*.png" \
        -c:v libx264 -pix_fmt yuv420p "$output_dir/$vid_name"
    echo "    ✓ Video: $vid_name"

    # 2. Extract branch JSON
    python3 "$SIM_DIR/extract_json.py" \
        --input "$MAIN_JSON" \
        --output "$output_dir/$json_name" \
        --scenario "$json_scenario" \
        --branch "$branch"

    # 3. Validate overlay (BEV)
    python3 "$SIM_DIR/validate_overlay.py" \
        -i "$output_dir/$vid_name" \
        -o "$output_dir/$validated_name" \
        -j "$output_dir/$json_name" \
        -t "$target_key"
}

# --- Function to process Test folder (pre-intervention) ---
process_test() {
    local raw_sc_dir="$1"       # e.g. /workspace/out/scenario_1
    local first_branch="$2"     # e.g. Factual_AEB (use frames from the first branch)
    local output_dir="$3"       # e.g. .../output/scenario_1/test
    local json_scenario="$4"    # e.g. high_speed_following
    local target_key="$5"       # e.g. npc

    mkdir -p "$output_dir"

    # 1. FFmpeg: join first PRE_FRAMES frames → test video
    local frames_dir="$raw_sc_dir/$first_branch"
    ffmpeg -loglevel warning -y -framerate $FPS -pattern_type glob \
        -i "$frames_dir/*.png" \
        -frames:v $PRE_FRAMES \
        -c:v libx264 -pix_fmt yuv420p "$output_dir/test.mp4"
    echo "    ✓ Test Video: test.mp4 ($PRE_FRAMES frames)"

    # 2. Extract pre-intervention JSON (time_sec < INTERVENTION_TIME)
    python3 "$SIM_DIR/extract_json.py" \
        --input "$MAIN_JSON" \
        --output "$output_dir/test.json" \
        --scenario "$json_scenario" \
        --before-time $INTERVENTION_TIME

    # 3. Validate overlay (BEV) for test video
    python3 "$SIM_DIR/validate_overlay.py" \
        -i "$output_dir/test.mp4" \
        -o "$output_dir/validated_test.mp4" \
        -j "$output_dir/test.json" \
        -t "$target_key"
}

# ═══════════════════════════════════════════
# SCENARIO 1: HIGH-SPEED FOLLOWING
# ═══════════════════════════════════════════
echo -e "\n--- Scenario 1: High-Speed Following ---"

S1_RAW="$RAW_DIR/scenario_1"
S1_OUT="$OUTPUT_DIR/scenario_1"
S1_JSON_NAME="high_speed_following"
S1_TARGET="npc"

for branch in "Factual_AEB" "Counterfactual_Distracted" "Counterfactual_Swerve"; do
    echo "  -> Intervention: $branch"
    process_intervention "$S1_RAW" "$branch" "$S1_OUT/interventions/$branch" "$S1_JSON_NAME" "$S1_TARGET"
done

echo "  -> Test (pre-intervention)"
process_test "$S1_RAW" "Factual_AEB" "$S1_OUT/test" "$S1_JSON_NAME" "$S1_TARGET"

# ═══════════════════════════════════════════
# SCENARIO 2: PEDESTRIAN CROSSING
# ═══════════════════════════════════════════
echo -e "\n--- Scenario 2: Pedestrian Crossing ---"

S2_RAW="$RAW_DIR/scenario_2"
S2_OUT="$OUTPUT_DIR/scenario_2"
S2_JSON_NAME="pedestrian_crossing"
S2_TARGET="walker"

for branch in "Factual_Yield" "Counterfactual_Panic_Brake" "Counterfactual_Swerve"; do
    echo "  -> Intervention: $branch"
    process_intervention "$S2_RAW" "$branch" "$S2_OUT/interventions/$branch" "$S2_JSON_NAME" "$S2_TARGET"
done

echo "  -> Test (pre-intervention)"
process_test "$S2_RAW" "Factual_Yield" "$S2_OUT/test" "$S2_JSON_NAME" "$S2_TARGET"

# ═══════════════════════════════════════════
# PHASE 3: BUILD GROUND TRUTH
# ═══════════════════════════════════════════
echo -e "\n[Phase 3] Building ground truth..."
python3 "$SIM_DIR/build_ground_truth.py" \
    --input "$MAIN_JSON" \
    --output-dir "$OUTPUT_DIR"

# ═══════════════════════════════════════════
# PHASE 4: COPY GT TO DATA/ (for version control)
# ═══════════════════════════════════════════
echo -e "\n[Phase 4] Copying ground truth to data/ground_truth/..."
mkdir -p "$GT_DIR"
for gt_file in "$OUTPUT_DIR"/scenario_*/ground_truth.json; do
    scenario_dir=$(basename "$(dirname "$gt_file")")
    # scenario_1 → scenario_001
    scenario_num=$(echo "$scenario_dir" | grep -o '[0-9]*')
    dest_name=$(printf "scenario_%03d.json" "$scenario_num")
    cp "$gt_file" "$GT_DIR/$dest_name"
    echo "  ✓ $dest_name"
done

# ═══════════════════════════════════════════
# PHASE 5: CLEANUP INTERMEDIATES
# ═══════════════════════════════════════════
echo -e "\n[Phase 5] Cleaning up intermediates..."
rm -f "$MAIN_JSON"
echo "  ✓ Removed coriva_bench_examples.json"

# ═══════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════
echo -e "\n========================================="
echo " ALL DONE! 🎉"
echo "========================================="
echo ""
echo "Output structure:"
find "$OUTPUT_DIR" -type f | sort | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  ${f#$OUTPUT_DIR/}"
done
echo ""
echo "Ground truth (versioned):"
ls -lh "$GT_DIR"/*.json 2>/dev/null || echo "  (none)"
