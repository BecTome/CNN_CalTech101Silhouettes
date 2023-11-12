# Script to initialize the project
INPUT_DIR="input/data/"
RAW_DIR="input/data/raw"
PARTITION_SCRIPT="helpers/partitions.py"

VAL_RATIO1=0.1
TEST_RATIO1=0.1

VAL_RATIO2=0.2
TEST_RATIO2=0.4

VAL_RATIO3=0.1
TEST_RATIO3=0.8

# Download the dataset
rm -rf $INPUT_DIR
mkdir -r $RAW_DIR
wget -P $RAW_DIR https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat

# Create partitions

python3 $PARTITION_SCRIPT --input_path "${RAW_DIR}/caltech101_silhouettes_28_split1.mat" \
                            --output_path $INPUT_DIR \
                            --val_ratio $VAL_RATIO1 \
                            --test_ratio $TEST_RATIO1
            
python3 $PARTITION_SCRIPT --input_path "${RAW_DIR}/caltech101_silhouettes_28_split1.mat" \
                            --output_path $INPUT_DIR \
                            --val_ratio $VAL_RATIO2 \
                            --test_ratio $TEST_RATIO2

python3 $PARTITION_SCRIPT --input_path "${RAW_DIR}/caltech101_silhouettes_28_split1.mat" \
                            --output_path $INPUT_DIR \
                            --val_ratio $VAL_RATIO3 \
                            --test_ratio $TEST_RATIO3
                                 

