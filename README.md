## BERT FOR Attribute Value Extraction

A Pytorch implementation of "Scaling Up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title" (ACL 2019).

paper: https://www.aclweb.org/anthology/P19-1514.pdf

### Structure of the model

The author explicitly model the semantic representations for attribute and title, and develop an attention mechanism to capture the interactive semantic relations in-between to enforce our framework to be attribute comprehensive.

![](./outputs/model.png)

Architecture of the proposed attribute-comprehension open tagging model

### requirement

1. pytorch=1.3.0
2. cuda=9.0

### data

```text
{"id": 19, "title": "热风2019年春季新款潮流时尚男士休闲皮鞋透气低跟豆豆鞋h40m9107", "attribute": "款式", "value": "豆豆鞋"}
```
download link: [BaiDuYun](https://pan.baidu.com/s/1pChOPH0bShN5elcH2rSbcw)

### How to use the code

1. Modify the configuration information in `run_attr_crf.py` or `run_attr_crf.sh` .
2. `sh run_attr_crf.sh`

### result

coming soon