# L-DYNO: Framework to Learn Consistent Visual Features Using Robot’s Motion.
This is an official implementation of our work published in ICRA'24. The [paper](https://arxiv.org/abs/2310.06249) utilizes the robot's transformations through an external signal (inertial sensing, for example) and gives attention to image space that is most consistent with the external signal.
# Training:
```python
train.py
```
> We train the pipeline end-to-end on the RTX A100.
graphics card
# Mask Generation:
```python
viz_masked.py
```
## To test the generated masks on trajectory estimation and reprojection errors follow this [repo](https://github.com/luigifreda/pyslam).
https://github.com/kartikeya13/L-DYNO/assets/36641341/8516b6ec-70d5-48a1-890b-90f28d2d0bae

