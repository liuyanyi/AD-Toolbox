## AD-Toolbox

Aerial Detection Toolbox based on MMDetection and MMRotate.

Mainly for my own convenience, no need to modify code in mmdet and mmrotate.

English | [简体中文](README_zh-CN.md)

## Supported Datasets

| Dataset  | HBB Det | OBB Det | Instance Segm |
|:--------:|:-------:|:-------:|:-------------:|
|   DOTA   |    ×    |   ✓ *   |       -       |
| HRSC2016 |    ×    |   ✓ *   |       -       |
|   DIOR   |    ✓    |    ✓    |       -       |
|  VHR-10  |    ✓    |    ×    |       ✓       |
|  MAR20   |    ✓    |    ✓    |       -       |

**Noting:**

- `*` : `DOTA` and `HRSC2016` are officially support by MMRotate.

## Dependencies

- [MMCV-Full](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)

