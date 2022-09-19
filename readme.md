## Introduction

### Project Structure

```bash
.
├── datasets.py                # dataset & dataloader                     
├── multiple_adba.py     # entrypoint
├── nets                           # network definition
├── test_backdoor.py      # test effect of trigger
├── test_imagenet.py      # test effect of trigger for imagenet dataset
├── test.py                       # test utils
├── train.py                     # train procedure
└── utils.py                     # utils
```

### Run Example

```bash
python multiple_adba.py --exp_name imagenet_100_glr\=0.001_lm\=0.0001 -si 0 -ei 99 --dataset imagenet -bs 64 --lr 0.025 -i 5 -glr 0.001 -lm 0.0001 -me 2
```