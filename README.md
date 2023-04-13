### Progress


##### 04/07

Wrote a simple coarse-stage network (according to the SkiRT paper section 4.1)

- Used the vertices of the SMPL-X model as input. Note that the point distribution in this case is not consistent across the body.
- Used 8 layers of MLP. Input: point position (dim=3) + global geometric feature (dim=256). Using local geometric feature for every point may potentially improve the result.
- Trained a separate network for every garment. It should also work when all the garments are trained together using only one network.

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

The point cloud is too sparse using the vertices as input (and the density of point cloud is not consistent). Better approach: `sample_point_uniformly`.

We can sample arbitrarily dense point cloud from the SMPL-X mesh, and perform interpolation to get the LBS weights of the sampled points. The problem is that this greatly slows down the training.

**Next**

- 写Fine stage阶段的网络
- sparse point cloud -> dense point cloud: How?
- LBS weight field: is it really necessary?

##### 04/11

**Change** 

- Fix the sampling issue. Use `trimesh.sample.sample_surface_even` to sample evenly across the body surface.
- Adopted positional encoding and the results seemed to improve a little.
- Added repulsion loss. Note that the current result is trained without the repulsion loss. Adding this loss can possibly prompt the generated point cloud to distribute evenly.

**Result**

![image](./results/rp_felice_posed_004_vis/pred_pcd.gif)

**Problems**

The point cloud between two legs still looks sparse. The additional fine network may be able to mitigate this problem as it also takes pose as input and has a local geometry feature for every point (instead of a global geometry feature).