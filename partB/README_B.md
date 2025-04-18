# Part B: Fine-tuning Pre-trained CNN Model

This folder contains the implementation for fine-tuning a pre-trained CNN model on the iNaturalist dataset.

## Files and Descriptions
- `a2_partB.py`: Script to fine-tune pre-trained CNN (I have used ResNet50)
- `sweep_2.yaml`: Sweep configuration for fine-tuning hyperparameters.

## How to Run the Code

### Fine-tuning and Hyperparameter Tuning:
To run the main code:
```bash
python a2_partB.py --data_dir <path_to_dataset> --max_epochs 10 --batch_size --freeze_stratergy <refer_report> --lr --weight_decay --image_size --data_augment
```
WandB Hyperparameter Sweep:
```bash
wandb sweep sweep_2.yaml
wandb agent <sweep_id>
```
## Details about the freezing stratergies:

The strategies I tried are mentioned below:
1. Freezing all layers except the final layer (Strat1): Reduce computation significantly by updating only the classification head.
2. Freezing all layers except last 2 layers (Strat2): Balance between efficiency and allowing meaningful model updates.
3. Full fine-tuning (Strat3): Fine-tuning the entire network, typically when ample computational resources and data are available

I have tried the second strategy (strat2) as it strikes a good balance between training efficiency and accuracy, making it ideal for limited computational resources. 
