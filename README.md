# SiMFy

Code and dataset of the paper

[SiMFy: A Simple Yet Effective Approach for Temporal Knowledge Graph Reasoning](https://aclanthology.org/2023.findings-emnlp.249/)


## Preprocess
```bash
python preprocess.py --dataset ICEWS14
```

## Train
```bash
python train.py --dataset ICEWS14 --gpu 0 --embedding_dim 200 --learning_rate 0.001  --max_epochs 30 --alpha 0.001 --k 1 
```

## Evaluation
```bash
python train.py --dataset ICEWS14 --gpu 0 --embedding_dim 200 --learning_rate 0.001  --max_epochs 30 --alpha 0.001 --k 1 --test --metric raw --multi_step
```

## Citation

If you find this project useful in your research, please cite the following paper

```bibtex
@inproceedings{liu-etal-2023-simfy,
    title = "{S}i{MF}y: A Simple Yet Effective Approach for Temporal Knowledge Graph Reasoning",
    author = "Liu, Zhengtao and Tan, Lei and Li, Mengfan and Wan, Yao and Jin, Hai and Shi, Xuanhua",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
    pages = "3825--3836",
}
```
## Support or Contact
SiMFy is developed in the National Engineering Research Center for Big Data Technology and System, Cluster and Grid Computing Lab, Services Computing Technology and System Lab, School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China by Zhengtao Liu(zhengtaoliu@hust.edu.cn), Lei Tan(tanlei@hust.edu.cn), Mengfan Li(limf@hust.edu.cn), Yao Wan(wanyao@hust.edu.cn), Hai Jin(hjin@hust.edu.cn), Xuanhua Shi(xhshi@hust.edu.cn).
