import wandb
import os

project_name = os.getenv("PROJECT_NAME", "default_project")

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_pr_auc_macro",
        "goal": "maximize"
    },
    "parameters": {
        "num_layers": {"values": [1, 2, 3, 4]},
        "heads": {"values": [1, 2, 4]},
        "dropout": {"values": [0.2, 0.3]},
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
        "weight_decay": {"values": [1e-4, 5e-4]},
        "batch_size": {"values": [128, 256]},
        "neighbors": {"values": [5, 10, 15, 20]},
        "masking_ratio": {"values": [0.15, 0.2, 0.3]},
        "norm_type": {"values": ["layernorm"]},
        "global_skip":  {"values": [False]},
        "use_residual": {"values": [True, False]},
        "emb_dim": {"values": [256, 512, 1024]},
    }
}

sweep_id = wandb.sweep(sweep_config, project=project_name, entity="aa8417-princeton-university")

with open(f"./sweep_ids/sweep_id_{project_name}.txt", "w") as f:
    f.write(sweep_id)

print("Sweep ID:", sweep_id)