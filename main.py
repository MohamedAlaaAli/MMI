from trainer import Trainer
import wandb

wandb.login(key="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")

if __name__ == "__main__":

    trainer = Trainer(config_path="configs/unet.json")
    trainer.fit()
