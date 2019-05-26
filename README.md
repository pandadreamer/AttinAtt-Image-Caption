# AttinAtt-Image-Caption

## Requirements
- Python 2.7
- Pytorch 0.4

## Train our network on COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

Then Get `data/cocotalk.json` and `data/cocotalk_label.h5`

### Get Improved Bottom-up from caffe

Put three datasets `train2014_resnet101_faster_rcnn_genome.tsv`, `test2014_resnet101_faster_rcnn_genome.tsv` and `val2014_resnet101_faster_rcnn_genome.tsv` into `data/bu_data/trainval/`, 

Then:

```bash
$ python scripts/make_bu_data.py --output_dir data/cocobu
```

So this will create `data/coco_fc`, `data/cocobu_att` and `data/cocobu_box`, which contains fc, features and boxes.

### Start training

```bash
$ python train.py --id topdown --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_topdown --save_checkpoint_every 2000 --val_images_use 5000 --max_epochs 50  
```

In my methods, this train result will be stored in `--checkpoint_path`, which is `log_topdown`, so we can continue train the model from this floder.

`--language_eval 1` means using CIDEr to replace cross entropy loss, default value is 0.


### Using self-critial to train

First preprocess the dataset and get the cache for calculating cider score:

```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

Then store scst in another floder `log_topdown_rl`

```
$ bash scripts/copy_model.sh topdown topdown_rl
```

Finally:

```bash
$ python train.py --id topdown_rl --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_topdown_rl --checkpoint_path log_topdown_rl --save_checkpoint_every 2000 --language_eval 1 --val_images_use 5000 --self_critical_after 50 --beam_size 2
```

You can see all CIDEr scores each epoch in `train_log_tiodown_rl`

## Evaluate

### Evaluate on Karpathy's test split

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model log_topdown_rl/model-best.pth --infos_path log_topdown_rl/infos_topdown_rl-best.pkl --language_eval 1 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att
```


The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.
