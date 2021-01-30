# To Train 

1. Download Bert pretrained weights from [Pretrained-Bert](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 

2. Rename it from `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin` and place it into `/pybert/pretrain/bert/base-uncased` directory.

3. Download Pseudo-text dataset auto-generated from random auto-generated attributes from [Scaled Attributes](https://drive.google.com/file/d/1aCCfAqYdAvonhUDU9VhsDmfuXxe7SOuc/view?usp=sharing)

4. Place it into `/pybert/dataset` directory.

5. Install transformers
```bash
    pip install pytorch-transformers
```

6. Preprocess data.
```bash
    python run_bert.py --do_data
```

7. Fine tune Bert model
```bash
    python run_bert.py --do_train --save_best --do_lower_case
```

8. Complete Fine-Tuning from the last epoch, you can also specify the number of epochs
```bash
    python run_bert.py --do_train --save_best --do_lower_case --resume_from_last_trial --epochs n_epochs
```


# To Test

1. Place a csv file named as `test.csv` with the last column containing descriptions you want to test into `/pybert/dataset` directory.

2. Run
```bash
    python run_bert.py --do_test --do_lower_case
```

3. Check `test_results.csv` file to see the test_results



# To use it as a sub-module in your work
1. Download weights and config file from [Trained-Bert](https://drive.google.com/drive/folders/1Y0ViCkgwaEbLHt_ESzKOjDiISNjXsy3r?usp=sharing) 

2. Unzip it and place its contents into `/pybert/output/checkpoints/bert` directory.

4. Download `attributes_max.pkl` from [here](https://drive.google.com/file/d/1ul2m0t0-3xqglfz2neVnEUR5V-zT1Slb/view?usp=sharing), and place it in `scale_bert` folder.

3. Import `inference.py` file and use `bertMultiLabelClassifier` function to predict labels.
