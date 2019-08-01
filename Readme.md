###Requirements
```
python3.6
pytorch 1.0
```

###生成数据
```
python ./scripts/prepare_data.py --input ../data/kobe/item_desc_dataset.txt --outdir ../data/deepsegment/data/baseline --min_freq 5
```

###运行DEMO
```bash
python train.py --config configs/test.yaml --expname test
```
