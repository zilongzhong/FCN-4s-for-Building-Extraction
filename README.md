# Fully convolutional networks for building and road extraction

This is the reference implementation of the codes and models for the FCN-4s in the [IGARSS2016 FCN-4s paper](paper/FCN for building and road extraction.pdf).<br />

```jason
Fully convolutional networks for building and road extraction: Preliminary results
Zilong Zhong*, Jonathan Li, Weihong Cui, Han Jiang
IGARSS 2016
```

Futher exploration about FCN models please refer to [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org).<br />

```jason
Fully Convolutional Models for Semantic Segmentation
Jonathan Long*, Evan Shelhamer*, Trevor Darrell
CVPR 2015
arXiv:1411.4038
```
Before deploying this model, make sure you have installed the latest [BVLC/caffe:master](https://github.com/BVLC/caffe).<br />
[FCN-4s Model](caffemodel/FCN4s_building.caffemodel) for building extraction in high spatial resolution imagery.

Training command:<br />
```python
python TrainingVal/solve.py
```
Evaluating command:<br />
```python
python evaluate.py
```
