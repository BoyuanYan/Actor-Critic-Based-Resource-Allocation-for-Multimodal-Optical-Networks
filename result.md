## 波长维度扩展（before 0326）

|cluster|wave_num|network|network_factor|rou|miu|base_lr|max_iter|end|result|best now|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|local|5|simplestnet||8|120|7e-4|300|True||0.10|
|20|5|simplenet||8|120|7e-4|300|True|0.10|0.10
|20|10|simplenet||8|300|7e-4|3000|False|||
|20|10|expandsimplenet|expand=2|8|300|2e-4|3000|False||0.21|
|20|10|expandsimplenet|expand=4|8|300|7e-4|3000|False||0,33|
|20|10|expandsimplenet|expand=2|8|300|7e-5|3000|False||0.35|
|30|15|expandsimplenet|expand=4|8|480|1e-3|3000|False||0.4|
|30|10|expandsimplenet|expand=4|8|300|2e-4|3000|False||0.25|



|cluster|wave_num|expand_factor|base_lr|work?|lowest bp|finished?|
|:----|:----|:----|:----|:----|:----|:----|
|20|10|2|2e-4|Yes|0.154|Yes|
|20|10|2|7e-4|No|0.34|Yes|
|20|10|2|7e-5|Yes|0.18|Yes|
|20|10|3|2e-4|Yes|0.107|Yes|
|20|10|3|7e-4|Yes|0.215|Yes|
|20|10|4|7e-4|No(5M以后就跑飞了)|0.25|Yes|
|20|15|3|7e-4|No|0.320|Yes|
|20|15|3|2e-4|Yes|0.225|Yes|
|20|15|3|1e-4|Yes|0.25|Yes (8.5M)|
|20|20|3|7e-4|Yes|0.292|Yes|
|20|25|3|7e-4|Yes|0.272|Yes (7.2M)|
|20|30|3|7e-4|Yes|0.259|Yes(6.4M)|
|20|40|3|7e-4|Yes|0.226|Yes(5M)|
|30|10|4|1e-3|No|0.345|Yes|
|30|10|4|2e-4|Yes|0.11|Yes|
|30|15|4|1e-3|No|0.318|Yes (8.5M)|
|30|15|4|2e-4|Yes|0.168|**No** (11.1M)|
|30|20|4|2e-4|Yes|0.288|Yes (5.8M)|

此外，16集群上还有几个顽强存在，没有被扑杀的进程，但是上面运行的版本太老，结果不足以才信。以后16集群上不会再跑任务了。



结论：
1.

## 拓扑维度扩展

详情可见[2018ecoc.md](2018ecoc.md)

