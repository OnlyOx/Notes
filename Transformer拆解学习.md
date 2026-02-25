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
i： 0-2，用2i或2i+1覆盖整个维度，对应0-4

![](assets/Transformer拆解学习/file-20260225113655251.png)

PE(pos,2i)=sin()
PE(pos,2i+1)=cos()

pos=0时,对应 "我" 这个字
i=0,2i=0     ,PE=sin0=0
i=0,2i+1=1,PE=cos0=1
i=1,2i=2     ,PE=sin0=0
i=1,2i+1=3,PE=cos0=1
i=2,2i=4     ,PE=sin0=0

“我”的位置编码为[0 1 0 1 0]

pos=1时,对应 "爱" 这个字
i=0,2i=0     ,PE=sin1≈0.84
i=0,2i+1=1,PE=cos1=1
i=1,2i=2     ,PE=sin1=0
i=1,2i+1=3,PE=cos1=1
i=2,2i=4     ,PE=sin1=0