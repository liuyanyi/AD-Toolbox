# MAR20 Dataset

|      Model      | Pretrain |    Size     | mAP  | AP50 | AP75 | FPS  | speed (ms) |
|:---------------:|:--------:|:-----------:|:----:|:----:|:----:|:----:|:----------:|
| Rotated YOLOX-s |   COCO   | (1024,1024) | 60.7 | 88.3 | 73.5 | 37.2 |  26.9-T4   |
| Rotated YOLOX-s |    -     | (1024,1024) | 52.6 | 85.1 | 58.3 | 37.7 |  26.6-T4   |

**Note**:

- All models are trained with batch size 8 on one GPU.
- FPS and speed are tested on a single Tesla T4.
