
# Focal Inverse Distance Transform Map
* An officical implementation of Focal Inverse Distance Transform Map. We propose a novel map named Focal Inverse Distance Transform (FIDT) map,  which can represent each head location information.

## Overview
![avatar](./image/overview.png)

# Visualizations
![avatar](./image/fidtmap.png)

# Progress
- [x] Testing Code (2021.3.16)
- [x] Pretrained model
  - [x] ShanghaiA  (2021.3.16)
  - [x] ShanghaiB  (2021.3.16)

# Environment

python >=3.6
pytorch >=1.4
opencv-python >=4.0
scipy >=1.4.0
h5py >=2.10
pillow >=7.0.0
imageio >=1.18
nni >=2.0 (python3 -m pip install --upgrade nni)

# Datasets

- Download ShanghaiTech dataset from [Baidu-Disk](https://pan.baidu.com/s/15WJ-Mm_B_2lY90uBZbsLwA), passward:cjnx; or [Google-Drive](https://drive.google.com/file/d/1CkYppr_IqR1s6wi53l2gKoGqm7LkJ-Lc/view?usp=sharing)
- Download UCF-QNRF dataset from [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- Download JHU-CROWD ++ dataset from [here](http://www.crowd-counting.com/)
- Download NWPU-CROWD dataset from [Baidu-Disk](https://pan.baidu.com/s/1VhFlS5row-ATReskMn5xTw), passward:3awa; or [Google-Drive](https://drive.google.com/file/d/1drjYZW7hp6bQI39u7ffPYwt4Kno9cLu8/view?usp=sharing)

# Generate FIDT Ground-Truth

```
cd data
python fidt_generate_xx.py
```

“xx” means the dataset name, including sh, jhu, qnrf, and nwpu.

# Model

Download the pretrained model from [Baidu-Disk](https://pan.baidu.com/s/1SaPppYrkqdWeHueNlcvUJw), passward:gqqm or [Google-Drive](https://drive.google.com/drive/folders/1c-99hZaVaqIb7UV_8G0Dz4psuNncO5en?usp=sharing)

# Quickly test

- `git clone https://github.com/dk-liang/FIDTM.git`
  `cd AutoScale`
- Download Dataset and Model
- Generate FIDT map ground-truth
- Generate images list (run `python make_npydata.py`)

- Test
  `python val.py --test_dataset ShanghaiA --pre ./model/ShanghaiA/model_best.pth --gpu_id 0`
  `python val.py --test_dataset ShanghaiB --pre ./model/ShanghaiB/model_best.pth --gpu_id 0`
  More config information is provided in `config.py`
# Evaluation localization performance
  `cd ./local_eval
  python A_gt_generate.py 
  python eval.py`

  We choose two thresholds (4, 8) for evaluation. 


# Training

The official training code is coming soon. 
Also, the training strategy is simple. You can replace the density map with the FIDT map in any regressors for training.




