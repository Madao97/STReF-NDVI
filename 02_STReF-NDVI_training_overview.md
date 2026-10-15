Algorithm: STReF-NDVI training overview

Inputs:
    Sentinel-1 feature sequences
    Sentinel-2 NDVI reference observations
    Observation-quality masks and timestamps

1. Load and check the aligned input data.
2. Prepare training and validation samples.
3. Construct model inputs from SAR observations
   and their temporal information.
4. Initialize the reconstruction model.
5. Repeat until the stopping criterion is met:
       a. Predict NDVI sequences for training samples.
       b. Calculate the training objective using valid labels.
       c. Update model parameters.
       d. Evaluate predictions on validation samples.
       e. Save the best-performing checkpoint.
6. Export the selected checkpoint and evaluation records.

Outputs:
    Trained model checkpoint
    Validation metrics
