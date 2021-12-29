# Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction
The pytorch implementation for the paper of 'Cascaded Residual Attention Enhanced Road Extraction from Remote Sensing Images'

# Abstract
Efficient and accurate road extraction from remote sensing imagery is important for applications related to navigation and Geographic Information System updating. Existing data-driven methods based on semantic segmentation recognize roads from images pixel by pixel, which generally uses only local spatial information and causes issues of discontinuous extraction and jagged boundary recognition. To address these problems, we propose a cascaded attention-enhanced architecture to extract boundary-refined roads from remote sensing images. Our proposed architecture uses spatial attention residual blocks on multi-scale features to capture long-distance relations and introduce channel attention layers to optimize the multi-scale features fusion. Furthermore, a lightweight encoder-decoder network is connected to adaptively optimize the boundaries of the extracted roads. Our experiments showed that the proposed method outperformed existing methods and achieved state-of-the-art results on the Massachusetts dataset. In addition, our method achieved competitive results on new benchmark datasets, e.g., the DeepGlobe and the Huawei Cloud road extraction challenge.

# Citation
Li, S.; Liao, C.; Ding, Y.; Hu, H.; Jia, Y.; Chen, M.; Xu, B.; Ge, X.; Liu, T.; Wu, D. Cascaded Residual Attention Enhanced Road Extraction from Remote Sensing Images. ISPRS Int. J. Geo-Inf. 2022, 11, 9. https://doi.org/10.3390/ijgi11010009

```
@Article{ijgi11010009,
AUTHOR = {Li, Shengfu and Liao, Cheng and Ding, Yulin and Hu, Han and Jia, Yang and Chen, Min and Xu, Bo and Ge, Xuming and Liu, Tianyang and Wu, Di},
TITLE = {Cascaded Residual Attention Enhanced Road Extraction from Remote Sensing Images},
JOURNAL = {ISPRS International Journal of Geo-Information},
VOLUME = {11},
YEAR = {2022},
NUMBER = {1},
ARTICLE-NUMBER = {9},
URL = {https://www.mdpi.com/2220-9964/11/1/9},
ISSN = {2220-9964},
DOI = {10.3390/ijgi11010009}
}
```
