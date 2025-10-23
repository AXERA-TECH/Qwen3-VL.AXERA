# Qwen3-VL-4B-Instruct 视频理解耗时估算  

## input 
input video: 8 帧 384*384 

prefill token num 620
image token num 144*4 = 576

## 实测耗时  
image encoder: 
158 ms * 4 = 632ms 

prefill 阶段需要chunk 数 math.ceil(620/128) = 5

hidden layers = 36 

单层 prefill chunk 1 耗时: 11.1 ms 
单层 prefill chunk 2 耗时: 11.8 ms 
单层 prefill chunk 3 耗时: 12.3 ms 
单层 prefill chunk 4 耗时: 13.2 ms 
单层 prefill chunk 5 耗时: 13.6 ms 


单层 prefill 5个chunk耗时： 11.1 + 11.8 + 12.3 + 13.2 + 13.6 = 62 ms
36层 prefill 5个chunk耗时：62 * 36 = 2232 ms 
post layer 耗时 20 ms 

prefill 总耗时： 2232 + 20 = 2252 ms

decode 单层耗时：6.0 ms
36 层 decode 耗时: 36*6 = 216 ms

一次 decode 总耗时：216 + 20 = 236 ms 

## 总结  
image encoder time: 632ms 
llm ttft: 2252 ms 
decode 速度： 1000/236 = 4.2 tokens/s

