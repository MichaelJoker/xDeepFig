The group members are:
Xu Bokai 1730014072
Li Xinyue  1730026048
Lin Yanzhi  1730026065
Bu Shihan 1730026002

Our group aims at implementing feature generation CNN (FGCNN) and xDeepFM for Click-through rate (CTR) prediction. In addition, we proposed our own idea by combining these two models together and achieved a better performance.

**ctr_predict.py**: The main python file for our project
**FGCNN_xDeepFM.py**: Our Proposed idea by combining these two models together
Also, there are three folders called **core**, **datasets** and **models**. **core** folder contains some necessary modules for implementing FGCNN, xDeepFM and our combined model. **datasets** folder contains our data set and **data_sample_processing.ipynb** for processing the raw data set as it is large and we cannot process it even in our server. So we need to sample it. **models** folder contains two baseline models, DCN and DeepFM, and it also contains FGCNN and xDeepFM models.

The dataset can be downloaded from https://pan.baidu.com/s/1PtnuIyXBWvPFis0m4r9IMg 
passcode：wr3h

Put the datasets under folder datasets

Paper: https://dl.acm.org/doi/abs/10.1145/3502300.3502306
