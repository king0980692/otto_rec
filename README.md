# Otto Recommendation 

## Get the Data

Original Data format is *jsonl* format, but jsonl take tool much time to process with,
so just use the processed data by kaggle's user.

### 0. Prepare

#### Prepare Kaggle cli tool and its credentials

1. check if `~/.kaggle/kaggle.json` exist
2. `pip install kaggle`


#### Create data folder
```
mkdir data
```


### 1. Train/Valid data

#### Get the data
```
kaggle datasets download -d cdeotte/otto-validation -p data/
unzip data/otto-validation.zip -d data/split_chunked_parquet/
```

This data contains:
* train
    * processed **training** data in chunked parquet format in `train_parquet` folder
* valid
    * processed **validation** data in chunked parquet format in `test_parquet` folder 


### 2. Train/Test data

#### Get the data
```
kaggle datasets download -d columbia2131/otto-chunk-data-inparquet-format -p data/
unzip data/otto-chunk-data-inparquet-format.zip -d data/chunked_parquet/

```
This data contains:
* train
    * processed **training** data in chunked parquet format in `train_parquet` folder
* test
    * processed **test** data in chunked parquet format in `test_parquet` folder 

### 3. Baseline score



