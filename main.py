from src.dataset import SteamAppsDataset
from src.models.distilbert_fine_tune import DistilBertFTTrainer
import omegaconf
from torch.utils.data import random_split, DataLoader
import torch


if __name__ == "__main__":
    db_params = omegaconf.OmegaConf.load("configs/db_params.yaml")
    cfg = omegaconf.OmegaConf.load("configs/distilbert_fine_tune.yaml")

    dataset = SteamAppsDataset(db_params, cfg.get("tags")) # type: ignore

    n_train_samples = int(len(dataset) * cfg["p_train_samples"]) # type: ignore
    rng = torch.Generator().manual_seed(cfg["random_seed"]) # type: ignore

    train, test = random_split(dataset, [n_train_samples, len(dataset)-n_train_samples], generator=rng)
    train_dataloader = DataLoader(train, batch_size=cfg.get('batch_size', 8), shuffle=True) # type: ignore
    test_dataloader = DataLoader(test, batch_size=cfg.get('batch_size', 8), shuffle=False) # type: ignore

    trainer = DistilBertFTTrainer(cfg, tags=dataset.tag_labels) # type: ignore
    trainer.train(train_dataloader, test_dataloader)
