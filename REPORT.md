# HMM Activity Recognition Results

## Group Members
- Mathias (OPPO CPH2641, 20Hz) - 28 recordings
- Daniel (iPhone 12 Pro, 32Hz) - 22 recordings

## Results Summary
- Total recordings: 50
- Test accuracy: 65.11%
- Unseen data accuracy: 39.3%

| Activity | Samples | Sensitivity | Specificity |
|----------|---------|-------------|-------------|
| Standing | 75 | 5.33% | 94.09% |
| Walking | 73 | 84.93% | 95.61% |
| Jumping | 64 | 78.12% | 98.13% |
| Still | 66 | 98.48% | 66.04% |

## Visualizations
- transition_matrix_20260308_135147.png
- confusion_matrix_20260308_135147.png
- activity_sequence_20260308_135147.png
EOFnb 
cat > REPORT.md << 'EOF'
# HMM Activity Recognition Results

## Group Members
- Mathias (OPPO CPH2641, 20Hz) - 28 recordings
- Daniel (iPhone 12 Pro, 32Hz) - 22 recordings

## Results Summary
- Total recordings: 50
- Test accuracy: 65.11%
- Unseen data accuracy: 39.3%

## Performance Metrics
| Activity | Samples | Sensitivity | Specificity |
|----------|---------|-------------|-------------|
| Standing | 75 | 5.33% | 94.09% |
| Walking | 73 | 84.93% | 95.61% |
| Jumping | 64 | 78.12% | 98.13% |
| Still | 66 | 98.48% | 66.04% |

## Visualizations Generated
- `transition_matrix_20260308_135147.png`
- `confusion_matrix_20260308_135147.png`
- `activity_sequence_20260308_135147.png`
- `state_probabilities_20260308_135147.png`

## Analysis
- **Easiest to distinguish**: Still vs Walking
- **Hardest to distinguish**: Standing vs Still
- **Baum-Welch converged at iteration 9**
