# LM-CIR

VLM-based critic re-ranking for composed image retrieval.

## Framework

![LM-CIR Framework](Framework.svg)

## Files

- `gpt_concise_caption.py`  
  GPT-based concise caption generation.

- `concise_retrieval.py`  
  CIR retrieval pipeline.

- `critic.py`  
  VLM critic for candidate re-ranking.

- `run_critic.py`  
  Run VLM critic re-ranking.

- `prompts.py`  
  Prompt templates.

## Performance

### FashionIQ Validation Benchmark  (ViT-L/14)

| DATASET | Shirt |  | Dress |  | Toptee |  | Avg. |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|METHOD| R@10 | R@50 | R@10 | R@50 | R@10 | R@50 | R@10 | R@50 |
| CIReVL | 29.49 | 47.40 | 24.79 | 44.76 | 31.36 | 53.65 | 28.55 | 48.57 |
| OSrCIR | 33.17 | 52.03 | 29.70 | 51.81 | 36.92 | 59.27 | 33.26 | 54.37 |
| AutoCIR | 34.00 | 53.43 | 24.94 | 45.81 | 33.10 | 55.58 | 30.68 | 51.60 |
| ImageScope | -- | -- | -- | -- | -- | -- | 31.36 | 50.78 |
| SEIZE | 33.04 | 53.22 | 30.93 | 50.76 | 35.57 | 58.64 | 33.18 | 54.21 |
| CoTMR | 35.43 | 54.91 | 31.18 | 55.04 | 38.55 | 61.33 | 35.05 | 57.09 |
| G-MIXER | 40.87 | 60.35 | 37.98 | 60.93 | 46.91 | **66.14** | 41.92 | 62.47 |
| Ours - Concise | 42.10 | 60.79 | 39.27 | 61.73 | 45.95 | 65.58 | 42.44 | 62.70 |
| Ours - Critic | **44.46** | **60.79** | **40.60** | **61.73** | **48.24** | 65.58 | **44.43** | **62.70** |

### CIRCO and CIRR Test Benchmarks (ViT-L/14)

| DATASET | CIRCO |  |  |  | CIRR |  |  |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|METHOD| mAP@5 | mAP@10 | mAP@25 | mAP@50 | R@1 | R@5 | R@10 | R@50 |
| CIReVL | 18.57 | 19.01 | 20.89 | 21.80 | 24.55 | 52.31 | 64.92 | 86.34 |
| OSrCIR | 23.87 | 25.33 | 27.84 | 28.97 | 29.45 | 57.68 | 69.86 | -- |
| AutoCIR | 24.05 | 25.14 | 27.35 | 28.36 | 31.81 | 61.95 | 73.86 | 92.07 |
| ImageScope | 28.36 | 29.23 | 30.81 | 31.88 | 39.37 | 67.54 | 78.05 | 92.94 |
| SEIZE | 24.98 | 25.82 | 28.24 | 29.35 | 28.65 | 57.16 | 69.23 | -- |
| CoTMR | 27.61 | 28.22 | 30.61 | 31.70 | 35.02 | 64.75 | 76.18 | 92.51 |
| IP-CIR | 26.43 | 27.41 | 29.87 | 31.07 | 29.76 | 58.82 | 71.21 | 90.41 |
| G-MIXER | 28.29 | 29.04 | 31.44 | 32.39 | 37.42 | 67.69 | 78.58 | -- |
| Ours - Concise | 29.01 | 29.62 | 32.05 | 33.17 | 38.39 | 69.23 | 80.17 | 94.27 |
| Ours - Critic | **39.82** | **39.61** | **40.74** | **41.86** | **47.93** | **77.37** | **84.72** | **94.27** |
## Data Availability

The generated concise captions and retrieval features used in this project will be publicly released after the associated paper is accepted.
