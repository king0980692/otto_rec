set -xe

python3 code/gen_w2v_pl.py \
    --train_dir data/chunked_parquet/train_parquet/ \
    --test_dir data/chunked_parquet/test_parquet/ 
    
python3 code/rec_w2v.py \
    --test_dir data/chunked_parquet/test_parquet/ \
    --model_dir . \
    --out . 

python3 tools/evaluation.py --submission_dir ./
