# To Train 

1. Download Bert pretrained weights from [Pretrained-Bert](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 

2. Rename it from `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin` and place it into `/pybert/pretrain/bert/base-uncased` directory.

3. Download Pseudo-text dataset generated from CelebA attributes from [CelebA-Pseudo-Descriptions](https://drive.google.com/file/d/1tJHFDdvmugWAAcVR_84QZ61tXBB8QBPs/view)

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



# To Test

1. Place a csv file named as `test.csv` with the last column containing descriptions you want to test into `/pybert/dataset` directory.

2. Run
```bash
    run_bert.py --do_test --do_lower_case
```

3. Check `test_results.csv` file to see the test_results



# To use it as a sub-module in your work

1. Download weights and config file from [Trained-Bert](https://drive.google.com/file/d/1OI9GwSI3VSZ-AnaY4kKN6h-krTAPvg7n/view?usp=sharing) 

2. Unzip it and place its contents into `/pybert/output/checkpoints/bert` directory.

3. Import `inference.py` file and use `bertMultiLabelClassifier` function to predict labels.