# DRIDNet

DRIDNet is a clean release package for Houston2013 hyperspectral and LiDAR classification. The repository keeps Houston2013 ready to run and leaves placeholder folders for Houston2018, MUUFL, and Trento.

## Environment

Recommended environment:

```bash
conda create -n dridnet python=3.10 -y
conda activate dridnet
pip install -r requirements.txt
```

Core dependencies are PyTorch, TorchVision, NumPy, SciPy, scikit-learn, einops, matplotlib, and imageio.

## Repository Layout

```text
DRIDNet/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── dsfem.py
│   ├── ciidm.py
│   ├── runner.py
│   └── metrics.py
├── data/
│   ├── Houston2013/
│   │   ├── Houston2013_hsi.mat
│   │   ├── Houston2013_lidar.mat
│   │   ├── Houston2013_gt.mat
│   │   └── Houston2013_index.mat
│   ├── Houston2018/.gitkeep
│   ├── Muufl/.gitkeep
│   └── Trento/.gitkeep
├── weights/
│   ├── Houston2013/
│   │   ├── intrinsic_decomposition.pth
│   │   └── spectral_prior_ddpm.pt
│   ├── Houston2018/.gitkeep
│   ├── Muufl/.gitkeep
│   └── Trento/.gitkeep
└── outputs/.gitkeep
```

## Data and Weights

Houston2013 is the only dataset included in this release:

- HSI: `data/Houston2013/Houston2013_hsi.mat`
- LiDAR/DSM: `data/Houston2013/Houston2013_lidar.mat`
- Ground truth: `data/Houston2013/Houston2013_gt.mat`
- Train/test index: `data/Houston2013/Houston2013_index.mat`
- Intrinsic/decomposition initialization: `weights/Houston2013/intrinsic_decomposition.pth`
- Spectral prior initialization: `weights/Houston2013/spectral_prior_ddpm.pt`

The other dataset folders are placeholders. Add matching `.mat` files and weights there before running those datasets.

## Run

Train Houston2013:

```bash
python main.py --dataset Houston2013 --mode train --gpu 0 --epochs 100
```

Test with a checkpoint produced by training:

```bash
python main.py --dataset Houston2013 --mode test --gpu 0 --test_ckpt outputs/Houston2013/<run_id>/ckpt/OA=<score>.pth
```

Generate a full-map prediction:

```bash
python main.py --dataset Houston2013 --mode full_pre --gpu 0 --test_ckpt outputs/Houston2013/<run_id>/ckpt/OA=<score>.pth
```

Common optional arguments:

```bash
python main.py --dataset Houston2013 --mode train --gpu 0 --batch_size 64 --epochs 100 --learning_rate 2e-4 --channels 19 --hsi_windowSize 11
```


