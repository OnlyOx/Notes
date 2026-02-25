# 一、词嵌入
用一个一行多列的向量
代替一个词
![](assets/Transformer拆解学习/file-20260225111001620.png)
# 二、QKV
## 1、Positional Encoding（位置编码）
![](assets/Transformer拆解学习/file-20260225111001619.png)
pos：这个字在句子中的第几位
d：嵌入维度
i：具体第几个维度

以上述 我爱中国 为例
pos：0-3
d: 5
i： 0-4