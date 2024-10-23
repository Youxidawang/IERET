# IERET
Implicit expression recognition enhanced table-filling for aspect sentiment triplet extraction. Yanbo Li, Qing He, Nisuo Du, Qingni He.
Neurocomputing.

The code is based on [BDTF-ABSA](https://github.com/HITSZ-HLT/BDTF-ABSA), and thanks them very much.

## Requirements

- transformers==4.15.0
- pytorch==1.13.0+cu117
- einops=0.4.0
- torchmetrics==0.7.0
- tntorch==1.0.1
- pytorch-lightning==1.3.5

## Usage
### Training
```
python aste_train.py --dataset fashion
```

## Citation
If you use the code in your paper, please kindly star this repo and cite our paper.
