# Part A: Training CNN from Scratch

This folder contains code for building and training a Convolutional Neural Network (CNN) from scratch on the iNaturalist dataset.

## Files and Descriptions
- `a2_partA.py`: Main script to build and train CNN from scratch.
- `a2_partA_q4.py`: Script specifically to evaluate the best model on the test set.
- `prediction_grid.png`: A visual grid showing predictions from the test set. Output of a2_partA_q4.py
- `sweep.yaml`: Hyperparameter sweep configuration used in WandB.

## How to Run the Code

### Training and Validation:
To run the main code:
```bash
python a2_partA.py --data_dir <path_to_dataset> --max_epochs <5/10/20> --batch_size <32/64>
```
To run the best model script:
```bash
python a2_partA_q4.py --data_dir <path_to_dataset> --max_epochs <5/10/20> --batch_size <32/64> --save_grid <name.png>
```

### Hyperparameter Tuning with WandB:
Ensure you have set up WandB credentials before running sweeps.

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```
