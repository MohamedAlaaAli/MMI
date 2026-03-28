from trainer import Trainer
import wandb

wandb.login(key="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")

if __name__ == "__main__":

    trainer = Trainer(config_path="configs/unet.json")
    trainer.fit()
