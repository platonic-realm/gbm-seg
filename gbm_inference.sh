#!/bin/bash

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name=*)
      NAME="${1#*=}"
      shift
      ;;
    --snapshot=*)
      SNAPSHOT="${1#*=}"
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --sample-dimension=*)
      SAMPLE_DIMENSION="${1#*=}"
      shift
      ;;
    --stride=*)
      STRIDE="${1#*=}"
      shift
      ;;
    --scale-factor=*)
      SCALE_FACTOR="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Required options:"
      echo "  --name=NAME             Project name"
      echo "  --snapshot=SNAPSHOT     Snapshot file (e.g. 000-0600.pt)"
      echo "  --batch-size=SIZE       Batch size for inference"
      echo "  --sample-dimension=DIM  Sample dimensions (e.g. '12, 256, 256')"
      echo "  --stride=STRIDE         Stride values (e.g. '12, 128, 128')"
      echo "  --scale-factor=FACTOR   Scale factor value"
      echo ""
      echo "Optional:"
      echo "  --help                  Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check for required parameters
if [ -z "$NAME" ] || [ -z "$SNAPSHOT" ] || [ -z "$BATCH_SIZE" ] || [ -z "$SAMPLE_DIMENSION" ] || [ -z "$STRIDE" ] || [ -z "$SCALE_FACTOR" ]; then
  echo "Error: Missing required parameters."
  echo "Use --help for usage information"
  exit 1
fi

# Create tag by removing spaces and commas from specific parameters
CLEAN_SAMPLE_DIMENSION=$(echo "$SAMPLE_DIMENSION" | tr -d " ," )
CLEAN_STRIDE=$(echo "$STRIDE" | tr -d " ," )

# Build the tag
TAG="${SNAPSHOT}_${CLEAN_SAMPLE_DIMENSION}_${CLEAN_STRIDE}_${SCALE_FACTOR}"

echo "Job parameters:"
echo "  Name: $NAME"
echo "  Snapshot: $SNAPSHOT"
echo "  Batch size: $BATCH_SIZE"
echo "  Sample dimension: $SAMPLE_DIMENSION"
echo "  Stride: $STRIDE"
echo "  Scale factor: $SCALE_FACTOR"
echo "  Generated tag: $TAG"
echo ""


# Submit jobs in sequence
echo "Submitting jobs in sequence..."

# Job 1: Inference
echo "Running: sbatch ./sbatch/infer.sbatch \"$NAME\" \"$SNAPSHOT\" \"$BATCH_SIZE\" \"$SAMPLE_DIMENSION\" \"$STRIDE\" \"$SCALE_FACTOR\""
JOB1=$(sbatch ./sbatch/infer.sbatch "$NAME" "$SNAPSHOT" "$BATCH_SIZE" "$SAMPLE_DIMENSION" "$STRIDE" "$SCALE_FACTOR" | awk '{print $4}')
echo "Submitted job 1 (Inference): $JOB1"

# Job 2: PSP
echo "Running: sbatch --dependency=afterok:$JOB1 ./sbatch/psp.sbatch \"$NAME\" \"$TAG\""
JOB2=$(sbatch --dependency=afterok:$JOB1 ./sbatch/psp.sbatch "$NAME" "$TAG" | awk '{print $4}')
echo "Submitted job 2 (PSP): $JOB2 (depends on $JOB1)"

# Job 3: Morph (parent job that spawns array)
echo "Running: sbatch --dependency=afterok:$JOB2 ./sbatch/morph.sbatch \"$NAME\" \"$TAG\""
JOB3=$(sbatch --dependency=afterok:$JOB2 ./sbatch/morph.sbatch "$NAME" "$TAG" | awk '{print $4}')
echo "Submitted job 3 (Morph): $JOB3 (depends on $JOB2)"

# Job 4: Blender (must wait for ALL morph array jobs to complete)
echo "Running: sbatch --dependency=aftercorr:$JOB3 ./sbatch/blender.sbatch \"$NAME\" \"$TAG\""
JOB4=$(sbatch --dependency=aftercorr:$JOB3 ./sbatch/blender.sbatch "$NAME" "$TAG" | awk '{print $4}')
echo "Submitted job 4 (Blender): $JOB4 (depends on ALL array tasks of $JOB3)"

# Job 5: Render (must wait for ALL blender array jobs to complete)
echo "Running: sbatch --dependency=aftercorr:$JOB4 ./sbatch/render.sbatch \"$NAME\" \"$TAG\""
JOB5=$(sbatch --dependency=aftercorr:$JOB4 ./sbatch/render.sbatch "$NAME" "$TAG" | awk '{print $4}')
echo "Submitted job 5 (Render): $JOB5 (depends on ALL array tasks of $JOB4)"

# Job 6: Export
echo "Running: sbatch --dependency=afterok:$JOB5 ./sbatch/export.sbatch \"$NAME\" \"$TAG\""
JOB6=$(sbatch --dependency=afterok:$JOB5 ./sbatch/export.sbatch "$NAME" "$TAG" | awk '{print $4}')
echo "Submitted job 6 (Export): $JOB6 (depends on $JOB5)"

echo ""
echo "All jobs submitted in sequence."
echo "Dependency chain: $JOB1 → $JOB2 → $JOB3 → $JOB4 → $JOB5 → $JOB6"
echo "Check status with: squeue -u \$USER"
echo "Check dependencies with: squeue -u \$USER -o \"%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R %E\""
