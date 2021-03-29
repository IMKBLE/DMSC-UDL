#DMSC-UDL: Deep Multi-view Subspace Clustering with Unified and Discriminative Learning#


This repo contains the source code and dataset for our TMM 2021 paper:
###Qianqian Wang, Jiafeng Cheng, Quanxue Gao, Guoshuai Zhao, Licheng Jiao, Deep Multi-view Subspace Clustering with Unified and Discriminative Learning, IEEE Transactions on Multimedia, 2020, 10.1109/TMM.2020.3025666.###

[Paper Link](https://ieeexplore.ieee.org/abstract/document/9204408/)  


**Bibtex**  

@article{Wang2020deepm,  
  title={ Deep Multi-view Subspace Clustering with Unified and Discriminative Learning},  
  author={Wang, Qianqian and Cheng, Jiafeng and Gao, Quanxue and Zhao, Guoshuai and Jiao, Licheng},  
  journal={IEEE Transactions on Multimedia},  
  volume={99},  
  number={9},  
  pages={1-11},  
  year={2021},   
}  
 
**DMSC-UDL Model:**

<div style="text-align: center; width: 900px; border: green solid 1px;">
<img src="./Images/fig1.jpg"  width="700"    title="Network Model" alt="Network Model" style="display: inline-block;"/>
<br></br>
<center></center>
</div>

Fig. 1. Multi-view clustering schematic diagram. The left half represents multi-view samples, and the right half represents shared connection matrices. Samples of the same color in view 1 and view 2 belong to the same cluster. The more concentrated the diagonal of the connection matrix is, the better the clustering effect is. Using only the global self-expression structure for each view to cluster, it is possible that the samples circled by red boxes with inaccurate weights are clustered into wrong clusters. Combining local structure and discriminative constraint with local graph structure for all views to improve the clustering effect can get a better clustering effect.  

<div style="text-align: center; width: 900px; border: green solid 1px;">
<img src="./Images/fig1.jpg"  width="700"    title="Network Model" alt="Network Model" style="display: inline-block;"/>
<br></br>
<center>Figure 1: Network Model</center>
</div>