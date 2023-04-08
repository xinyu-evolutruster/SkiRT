# SkiRT
SkiRT implementation

### Progress

##### 04/07

复现一个很简单的coarse stage网络。

- 暂时还是使用body vertices作为输入，虽然分布不均匀但是这样训练会比较快
- 8层MLP，输入：point position (dim=3) + global geometric feature (dim=256), 不知道给每个点使用local feature会不会效果比较好
- 每个类型的服装分别train一个network，合在一起应该也能work

**Parameters**

```
lr: 5e-5
lr_geomfeat: 1e-4
batch_size: 8
num_samples: 10475 (SMPL-X number of vertices)
epoch: 250
```

**Coarse stage: Results**

![image](./vis/pcl_anna.gif)
![image](./vis/pcl_felice.gif)

**Problems**

用vertices当输入，点云太稀疏了，sample points然后做插值也是可以的，但是太慢，而且似乎还有一些诡异的bug？想不出好的加速方法。

**Next**

- 写Fine stage阶段的网络
- sparse point cloud -> dense point cloud: How?
- LBS weight field: is it really necessary?