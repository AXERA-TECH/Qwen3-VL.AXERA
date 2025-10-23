# Qwen3-VL-2B-Instruct 视频理解耗时估算  

## input 
input video: 8 帧 384*384 

prefill token num 620
image token num 144*4 = 576

## 实测耗时  
image encoder: 
157 ms * 4 = 628ms 

prefill 阶段需要chunk 数 math.ceil(620/128) = 5

hidden layers = 28 

单层 prefill chunk 1 耗时: 5.8 ms 
单层 prefill chunk 2 耗时: 6.3 ms 
单层 prefill chunk 3 耗时: 6.5 ms 
单层 prefill chunk 4 耗时: 6.9 ms 
单层 prefill chunk 5 耗时: 7.3 ms 


单层 prefill 5个chunk耗时： 5.8 + 6.3 + 6.5 + 6.9 + 7.3 = 32.8 ms
28层 prefill 5个chunk耗时：32.8 * 28 = 918.4 ms 
post layer 耗时 16.2 ms 

prefill 总耗时：  918.4 + 16.2 = 934.6 ms

decode 单层耗时：3.2 ms
28 层 decode 耗时: 3.2 * 28 = 89.6 ms

一次 decode 总耗时：89.6 + 16.2 = 105.8 ms 

## 总结  
image encoder time: 628 ms 
llm ttft: 934.6 ms 
decode 速度： 1000/105.8 = 9.5 tokens/s

