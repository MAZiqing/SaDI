# SaDI (ICASSP 2023 paper)

Hengbo Liu, Ziqing Ma, Linxiao Yang, Tian Zhou, Rui Xia, Yi Wang, Qingsong Wen, Liang Sun, "SaDI: A Self-Adaptive Decomposed Interpretable Framework for Electricity Load Forecasting under Extreme Events", in Proc. IEEE 48th International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023), Rhodes Island, Greece, June 2023.

## Abstract
Accurate prediction of electric load is crucial in power grid planning and management. In this paper, we solve the electric load forecasting problem under extreme events such as scorching heats. One challenge for accurate forecasting is the lack of training samples under extreme conditions. Also load usually  changes dramatically in these extreme conditions, which calls for interpretable model to make better decisions. 
In this paper, we propose a novel forecasting framework, named Self-adaptive Decomposed Interpretable framework~(SaDI), which ensembles long-term trend, short-term trend, and period modelings to capture temporal characteristics in different components. The external variable triggered loss is proposed for the imbalanced learning under extreme events. 
Furthermore, Generalized Additive Model (GAM) is employed in
the framework for desirable interpretability. The experiments on both Central China electric load and public energy meters from buildings show that the proposed SaDI framework achieves
average $22.14\%$ improvement compared with the current state-of-
the-art algorithms in forecasting under extreme events in
terms of daily mean of normalized RMSE. 

Link of the paper will be released soon.


![image](https://user-images.githubusercontent.com/44238026/219562512-941333b9-a121-4d3f-ba61-898a016ec7ce.png)

## Overall framework of SaDI
<img src="https://user-images.githubusercontent.com/44238026/219563089-b8220c34-1bbf-4f54-ab04-5cdde25d53df.png" width="500">


## Main results
![image](https://user-images.githubusercontent.com/44238026/219563198-4cc596fa-8572-4166-90ca-ff6a7b94a04b.png)

## Further Reading

* [Survey] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. "**Transformers in time series: A survey**." arXiv preprint arXiv:2202.07125 (2022). [[paper]](https://arxiv.org/abs/2202.07125)
* [Tutorial] Qingsong Wen, Linxiao Yang, Tian Zhou, Liang Sun, "**Robust Time Series Analysis and Applications: An Industrial Perspective**," in the 28th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2022), Washington DC, USA, Aug. 14-18, 2022. [[Website]](https://qingsongedu.github.io/timeseries-tutorial-kdd-2022/)
* Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin, "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting," in Proc. 39th International Conference on Machine Learning (ICML 2022), Baltimore, Maryland, July 17-23, 2022. [[paper](https://arxiv.org/abs/2201.12740)]


## Contact

If you have any question or want to use the code, please contact maziqing.mzq@alibaba-inc.com or yangyang.lhb@alibaba-inc.com.

## Acknowledgements
Citation of BDG2 Data-Set
Nature Scientific Data (open access)
Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x.
