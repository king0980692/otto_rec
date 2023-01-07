set -xe

#python3 code/gen_w2v_pl.py \
    #--worker 32 \
    #--train_dir data/split_chunked_parquet/train_parquet/ \
    #--test_dir data/split_chunked_parquet/test_parquet/ 
    
for _type in train valid test
do
    python3 code/rec_w2v.py \
        --group exp  \
        --mode train  \
        --embed ./exp/w2v.emb \
        --dim 50 \
        --out ./output 
done

#python3 tools/evaluation.py \
    #--submission_dir ./output \
    #--model w2v \
    #--valid_labels ./data/split_chunked_parquet/test_labels.parquet 
