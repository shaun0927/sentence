Runpod 환경 사용법(A100 SXM x 4)

#venv 환경 활성화
source .venv/bin/activate

#vLLM 띄우기(Llama-3-Motif-102B)
git config --global credential.helper store
huggingface-cli login
>본인 토큰 입력
vllm serve "moreh/Llama-3-Motif-102B" --dtype bfloat16 -tp 4 --max-model-len 4096 --download-dir "/sentence" --port 8000

#Bert 모델 실험 결과(첫 문장 분류 성능)


monologg/koelectra-base-v3-discriminator
5-Fold CV summary:  avg F1 = 0.9031  ±  0.0033

klue/roberta-base
5-Fold CV summary:  avg F1 = 0.9129  ±  0.0025

klue/roberta-large
5-Fold CV summary:  avg F1 = 0.9285  ±  0.0041

kobert_base
5-Fold CV summary:  avg F1 = 0.8749  ±  0.0038

kobigbird-bert-base
5-Fold CV summary:  avg F1 = 0.9010  ±  0.0024

xlm-roberta-large
5-Fold CV summary:  avg F1 = 0.9119  ±  0.0023

kykim/bert-kor-base
5-Fold CV summary:  avg F1 = 0.9140  ±  0.0031

microsoft/infoxlm-large
약 0.88(prune)

microsoft/deberta-v3-large
 5-Fold CV summary:  avg F1 = 0.9118  ±  0.0032