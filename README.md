# Beyond the Prototype: Divide-and-conquer Proxies for Few-shot Segmentation

This repo contains the code for our **IJCAI 2022 Long Oral** [paper](http://arxiv.org/abs/2204.09903) "*Beyond the Prototype: Divide-and-conquer Proxies for Few-shot Segmentation*" by Chunbo Lang, Binfei Tu, Gong Cheng and Junwei Han.

> **Abstract:**  *Few-shot segmentation, which aims to segment unseen-class objects given only a handful of densely labeled samples, has received widespread attention from the community. Existing approaches typically follow the prototype learning paradigm to perform meta-inference, which fails to fully exploit the underlying information from support image-mask pairs, resulting in various segmentation failures, e.g., incomplete objects, ambiguous boundaries, and distractor activation. To this end, we propose a simple yet versatile framework in the spirit of divide-and-conquer. Specifically, a novel self-reasoning scheme is first implemented on the annotated support image, and then the coarse segmentation mask is divided into multiple regions with different properties. Leveraging effective masked average pooling operations, a series of support-induced proxies are thus derived, each playing a specific role in conquering the above challenges. Moreover, we devise a unique parallel decoder structure that integrates proxies with similar attributes to boost the discrimination power. Our proposed approach, named divide-and-conquer proxies (DCP), allows for the development of appropriate and reliable information as a guide at the "episode" level, not just about the object cues themselves. Extensive experiments on PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> demonstrate the superiority of DCP over conventional prototype-based approaches (up to 5~10% on average), which also establishes a new state-of-the-art.*

<p align="middle">
  <img src="figure/flowchart.jpg">
</p>

### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

   Please see [OSLSM](https://arxiv.org/abs/1709.03410) and [FWB](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.html) for more details on datasets. 

### Models

- Download the pre-trained backbones from [here](https://drive.google.com/file/d/1AQcvMHHpURZM67MMgV-S3T0Kz-h2q7FR/view?usp=sharing) and put them into the `DCP/initmodel` directory. 

### Usage

- Change configuration via the `.yaml` files in `DCP/config`, then run the `.sh` scripts for training and testing.

- **Stage1** *Meta-training*


  ```
  sh train.sh
  ```

- **Stage2** *Meta-testing*


  ```
  sh test.sh
  ```


### Performance

Performance comparison with the state-of-the-art approach (_i.e._, [PFENet](https://github.com/dvlab-research/PFENet)) in terms of **average** **mIoU** across all folds. 

1. ##### PASCAL-5<sup>i</sup>

   | Backbone | Method     | 1-shot                   | 5-shot                   |
   | -------- | ---------- | ------------------------ | ------------------------ |
   | VGG16    | PFENet     | 58.00                    | 59.00                    |
   |          | DCP (ours) | 61.31 <sub>(+3.31)</sub> | 65.84 <sub>(+6.84)</sub> |
   | ResNet50 | PFENet     | 60.80                    | 61.90                    |
   |          | DCP (ours) | 62.80 <sub>(+2.00)</sub> | 67.80 <sub>(+5.90)</sub> |

2. ##### COCO-20<sup>i</sup>

   | Backbone  | Method     | 1-shot                   | 5-shot                   |
   | --------- | ---------- | ------------------------ | ------------------------ |
   | ResNet101 | PFENet     | 38.50                    | 42.70                    |
   | ResNet50  | DCP (ours) | 41.39 <sub>(+2.89)</sub> | 46.48 <sub>(+3.78)</sub> |

### References

This repo is mainly built based on [PFENet](https://github.com/dvlab-research/PFENet), [SCL](https://github.com/zbf1991/SCL) and [SemSeg](https://github.com/hszhao/semseg). Thanks for their great work!

### To-Do List

- [x] Support different backbones
- [x] Multi-GPU training

## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@InProceedings{lang2022dcp,
    title={Beyond the Prototype: Divide-and-conquer Proxies for Few-shot Segmentation},
    author={Lang, Chunbo and Tu, Binfei and Cheng, Gong and Han, Junwei},
    booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2022}
}
```
