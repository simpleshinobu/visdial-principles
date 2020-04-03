# VisDial-principles

This repository is the updated PyTorch implementation for CVPR 2020 Paper "Two Causal Principles for Improving Visual Dialog", which is also the newest version for the Visual Dialog Challenge 2019 winner team (Here is the [report](https://drive.google.com/file/d/1fqg0hregsp_3USM6XCHx89S9JLXt8bKp/view)). For the detailed theories, please refer to our [paper](https://arxiv.org/abs/1911.10496). If you have any questions or suggestions, please email me (JIAXIN003@E.NTU.EDU.SG), (I do not usually browse my Github, so the reply to issues may be not on time).

Note that this repository is based on the official [code](https://github.com/batra-mlp-lab/visdial), for the newest official code, please refer to [vi-bert version](https://github.com/vmurahari3/visdial-bert#setup-and-dependencies).

If you find this work is useful in your research, please kindly consider citing:

```
@article{qi2019two,
  title={Two Causal Principles for Improving Visual Dialog},
  author={Qi, Jiaxin and Niu, Yulei and Huang, Jianqiang and Zhang, Hanwang},
  journal={arXiv preprint arXiv:1911.10496},
  year={2019}
}
```
### Setup and Dependencies
### usage
#### Preparing (download data)
#### Training
1.baseline (recommend to use checkpoint 5-7 to finetune)
```
python train_stage1_baseline.py --validate --in-memory --save-model
```
2.different loss functions for answer score sampling (dense finetuning, R3 as default, because of the dense samples are rare, the results maybe a little bit unstable). Besides, we add another newest loss function R4 (Normalized BCE, which is better than R2, recommended).
```
python train_stage2_baseline.py --loss-function R4 --load-pthpath checkpoints/baseline_withP1_checkpiont5.pth
```
3.question type implementation (download the qt file or create it follow our paper)

Besides, you can try to use question type preference to directly help baseline model to inference.

4.dictionary learning

Besides, is you think the MRR score is too low, you can try to fuse the output logits of baseline and finetuned baseline with the formula: sigmoid(logit(ft))*(sigmoid(logit(base))+0.2) (empirical formula) (this can keep MRR ~61 and NDCG ~72)

#### Evaluation
You can directly evaluate a model use the following code: (please check the settings in configs/evaluate.yml)
```
python evaluate.py --load-pthpath <the model checkpoint path>
```
If you have any other questions or suggestions, please kindly email me.
#### Acknowledgements

Thanks for the source code from [the official](https://visualdialog.org/)
