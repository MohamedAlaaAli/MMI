import torch
from trainer import Trainer
from analyze import ManifoldInteractionAnalyzer
import wandb

wandb.login(key="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")

if __name__ == "__main__":

    trainer_late = Trainer(config_path="configs/late_fusion.json")
    trainer_early = Trainer(config_path="configs/early_fusion.json")
    # train = input("Start training? (y/n): ")
    # if train.lower() == "y":
    #     trainer.fit()
    # else:
    print("Skipping training.")
    print("analyzing tangent spaces...")

    late_fusion_pth = "ckpts/late_fusion_unet.pt"
    early_fusion_pth = "ckpts/early_fusion_unet.pt"

    late_fusion_model  = trainer_late.model.load_state_dict(torch.load(late_fusion_pth)).to(trainer_late.device).eval()
    early_fusion_model = trainer_early.model.load_state_dict(torch.load(early_fusion_pth)).to(trainer_early.device).eval()

    # Extract features for all samples in the dataset
    ct_feats = []
    pet_feats = []
    fusion_feats = []
    for batch in trainer_late.val_loader:
        pet_in, ct_in, _ = batch
        with torch.no_grad():
            _, (pet_feat, ct_feat) = late_fusion_model(pet_in, ct_in)
            ct_feats.append(ct_feat.cpu().numpy())
            pet_feats.append(pet_feat.cpu().numpy())
            
            _, concat_feat = early_fusion_model(torch.cat([pet_in, ct_in], dim=1).to(trainer_early.device))

            
        
