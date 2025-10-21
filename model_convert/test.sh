# Qwen3-VL-4B-Instruct
export greedy='false'
export top_p=0.8
export top_k=20
export temperature=0.7
export repetition_penalty=1.0
export presence_penalty=1.5
export out_seq_length=16384

# Qwen3-VL-4B-Thinking
# export greedy='false'
# export top_p=0.95
# export top_k=20
# export repetition_penalty=1.0
# export presence_penalty=0.0
# export temperature=1.0
# export out_seq_length=40960

# python test.py
python run_image.py