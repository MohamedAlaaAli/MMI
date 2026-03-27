from trainer import Trainer


if __name__ == "__main__":
    trainer = Trainer(config_path="kaggle/working/MMI/configs/unet_ct_exp.json")
    trainer.fit()
