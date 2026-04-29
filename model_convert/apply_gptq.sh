export CUDA_VISIBLE_DEVICES=2

# need GPTQModel==6.0.3 , transformers==5.4.0

nohup python apply_gptq.py > apply_gptq.log &