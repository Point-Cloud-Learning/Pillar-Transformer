## Pillar Transformer: Is A Point Cloud Worth 10*10 Words for Point Cloud Recognition?

### Abstract

------------

> Inspired by the successful application of Vision Transformer (ViT) to 2D image recognition, we design Pillar Transformer (PiT) with a pure Transformer architecture for point cloud recognition, which verifies that a pure Transformer architecture can perform 3D vision tasks well and provides a theoretical possibility of exploring Transformer-based unified models in point cloud involved multi-modality analysis. PiT possesses a similar architecture to ViT, with pillarization and shape encoder at its core. The pillarization divides 3D space into a number of pillars with the same volume, forming a sequence of pillars, while the shape encoder encodes the local shape information consisting of points in each pillar, and these shape features are then passed through the standard Transformer Encoder with global modeling capability and a classification header to achieve point cloud recognition. PiT obtains competitive results on the most popular benchmark dataset ModelNet40, as well as on the challenging dataset ScanObjectNN with higher computational efficiency compared to other point cloud recognition networks.

### Pipeline

----------------

![image](https://github.com/Point-Cloud-Learning/Pillar-Transformer/assets/120387542/95e40d73-701e-4c01-8ccb-71df4c0d3293)


### Contributions


-----------------

> - Inspired by ViT, the paper proposes a pure Transformer architecture model for point cloud recognition, which is centered on the pillarization and shape encoder. The pillarization divides 3D space into a number of pillars with the same volume, forming a sequence of pillars, and the shape encoder encodes the local shape information consisting of points in each pillar, which is then fed into the standard Transformer Encoder to achieve point cloud recognition. The pillarization and the efficient operation of the shape encoder allow PiT to meet low computing overheads, and the use of the Transformer Encoder simultaneously empowers its global modeling capability. Besides, by adjusting the length and width of the base of a pillar our model can trade-off between accuracy and time overhead to adapt to different application scenarios.
>
> 

### Experimental results

---------------

|ModelNet40| | ScanObjectNN | |
|:--------:|:--------:|:---------:|:-------:|
|mAcc(%)|  90.2  | mAcc(%) | 81.0 |
|  OA(%)  |  93.6 |  OA(%)  | 82.3 |
|Inference(ms)| 32.1 | | |



