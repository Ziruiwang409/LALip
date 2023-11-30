# LALip
LALip: LLM-Aided Lip-Reading Sentences

## Quickstart

1. [Download](https://drive.google.com/file/d/1c87YxRnKmU6_xoy4kMpY87xhjQJ12rmX/view?usp=sharing) preprocessed dataset

2. Extract data.zip in the root projecct directory

3. Run train.py from the root project directory: 

### Troubleshooting

- Ensure that train.py data_path arg points to the correct direectory

### Notes

- The preprocessed dataset contains some invalid data points. These occurences are handled in the dataloader by ignoring these datapoints. If a batch size of 64 is set, and some batches contain less than 64, this is why.