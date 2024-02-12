# L-DYNO: Framework to Learn Consistent Visual Features Using Robot’s Motion.
This is an official implementation of our work published in ICRA'24. The [paper](https://arxiv.org/abs/2310.06249) utilizes the robot's transformations through an external signal (inertial sensing, for example) and gives attention to image space that is most consistent with the external signal.

## Datasets:
KITTI: Download the dataset (grayscale images) from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and prepare the KITTI folder as specified.

F1tenth Dataset: [TODO]
## Training:
```python
train.py
```
> We train the pipeline end-to-end on the RTX A100.
<img width="834" alt="Screenshot 2023-11-18 at 3 43 12 PM" src="https://github.com/kartikeya13/L-DYNO/assets/36641341/d7a46aae-bc47-44c7-943d-49baa7518809">

## Mask Generation:
```python
viz_masked.py
```
> To test the generated masks on trajectory estimation and reprojection errors follow this [repo](https://github.com/luigifreda/pyslam).

https://github.com/kartikeya13/L-DYNO/assets/36641341/8516b6ec-70d5-48a1-890b-90f28d2d0bae



