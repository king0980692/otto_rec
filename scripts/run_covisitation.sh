set -xe
# 1.
python3 code/gen_occur_rapid.py --mode valid --out exp &

# 2.
mkdir -p exp/group
kaggle datasets download -d adaubas/otto-valid-test-list -p exp
unzip exp/otto-valid-test-list.zip -d exp/group
rm exp/*.zip
# or
#python3 tools/gen_group.py \
    #--valid_dir ./data/split_chunked_parquet/test_parquet \
    #--test_dir ./data/chunked_parquet/test_parquet \
    #--out exp &


wait
# 3.
python3 code/rec_by_rapid.py --mode valid --pqt exp/ --group exp


# 4. evaluation
python3 tools/evaluation.py --submission_dir ./
