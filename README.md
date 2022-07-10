
# VisDial-principles

This repository is the updated PyTorch implementation for CVPR 2020 Paper "Two Causal Principles for Improving Visual Dialog", which is also the newest version for the Visual Dialog Challenge 2019 winner team (Here is the [report](https://drive.google.com/file/d/1fqg0hregsp_3USM6XCHx89S9JLXt8bKp/view)). For the detailed theories, please refer to our [paper](https://arxiv.org/abs/1911.10496). If you have any questions or suggestions, please email me (JIAXIN003@E.NTU.EDU.SG), (I do not usually browse my Github, so the reply to issues may be not on time).

Note that this repository is based on the official [code](https://github.com/batra-mlp-lab/visdial), for the newest official code, please refer to [vi-bert version](https://github.com/vmurahari3/visdial-bert#setup-and-dependencies).

If you find this work is useful in your research, please kindly consider citing:

```
@inproceedings{qi2020two,
  title={Two causal principles for improving visual dialog},
  author={Qi, Jiaxin and Niu, Yulei and Huang, Jianqiang and Zhang, Hanwang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10860--10869},
  year={2020}
}
```
#### Dependencies
create a conda environment:
```
conda env create -f environment.yml
```
download nltk:
```
python -c "import nltk; nltk.download('all')"
```
#### Preparing (download data and pretrained model)
1.download data and pretrained model respectively:

1.1.create directory data/ and download necessary files into the data/:

from the official [website](https://visualdialog.org/data):

[visdial_1.0_train.json](https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0)

[visdial_1.0_val.json](https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0)

[visdial_1.0_val_dense_annotations.json](https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0)

[visdial_1.0_train_dense_sample.json](https://www.dropbox.com/s/1ajjfpepzyt3q4m/visdial_1.0_train_dense_sample.json?dl=0)

1.2.from us or collect by yourself:(also save in the directory data/)

[features_faster_rcnn_x101_train.h5](https://drive.google.com/open?id=1eC80EMMEdZvWsKIl3YlEFpY4XHlvN9h8)

[features_faster_rcnn_x101_val.h5](https://drive.google.com/open?id=1_QoH-lbRCwPrcuiwVNjhW1yMxhqiLclB)

[features_faster_rcnn_x101_test.h5](https://drive.google.com/open?id=1hyMCJLXAyaNHmnoRZM8eF3fNia49oHLl)

[visdial_1.0_word_counts_train.json](https://drive.google.com/open?id=1zL8P5LnPzRbfaPxJXvFVGBlS7SumOB_g)

[glove.npy](https://drive.google.com/open?id=1y4oSqAwgu2gIcyuF5ZuMuNZ-c-89NGuJ)

[qt_count.json](https://drive.google.com/open?id=1hllnesIwb__kVHmn5Mtz9CLt9VXnCUS_)

[qt_scores.json](https://drive.google.com/open?id=1QlKy4lVHMlZ4hqw4tVaB608WMBo-eBDs) (the key in each question type is the index of candidate in answer list)

[100ans_feature.npy](https://drive.google.com/open?id=1vu9wMGc8GTj-83ILlUxyuk8_4aCLAIkm) (for initial answer dict)

1.3.download the pretrained model:

[baseline_withP1_checkpiont5.pth](https://drive.google.com/open?id=1LZizUL1lSnLU9ZPmePUfDDtSBQVjAyH8)

#### Training
0.1 Check your gpu id and change it at --gpu-ids

1.baseline (recommend to use checkpoint 5-7 to finetune)
```
python train_stage1_baseline.py --validate --in-memory --save-model
```
(Note: for other encoders, please follow the format and note in the code)

2.different loss functions for answer score sampling (dense finetuning, R3 as default, because of the dense samples are rare, the results maybe a little bit unstable). Besides, we add another newest loss function R4 (Normalized BCE, which is better than R2, recommended).
```
python train_stage2_baseline.py --loss-function R4 --load-pthpath checkpoints/baseline_withP1_checkpiont5.pth
```
3.question type implementation (download the qt file or create it follow our paper)
```
cd question_type
python train_qt.py --validate --in-memory --save-model --load-pthpath checkpoints/baseline_withP1_checkpiont5.pth
```
Note that you can train it from pretrained model or train it from scratch (adjust the lr and decay epochs carefully). Besides, you can try to use question type preference to directly help baseline model to inference. But here, note that the candidate answer list in training is different from the one in validation, please take care to do the index conversion.

4.dictionary learning (three steps: train dict, finetune dict, finetune the whole model)
```
cd hidden_dict
python train_dict_stage1.py --save-model
python train_dict_stage2.py --save-model --load-pthpath <pretrained dict from stage1>
python train_dict_stage3.py --load-dict-pthpath <pretrained dict from stage1> --load-pthpath checkpoints/baseline_withP1_checkpiont5.pth
```
Besides, after our code optimization, some implementations can get a little bit better results, but do not influence the conclusions of our principles. If you think the MRR score is too low, you can try train with larger one-hot weight to keep MRR. Furthermore, just use the top1 candidate of stage 1 model and the rest use ranks from finetuned model will get better balanced performance of both MRR and NDCG!

#### Evaluation
You can directly evaluate a model use the following code: (please check the settings in configs/evaluate.yml)
```
python evaluate.py --load-pthpath <the model checkpoint path>
```
If you have any other questions or suggestions, please kindly email me.
#### Acknowledgements

Thanks for the source code from [the official](https://visualdialog.org/)





