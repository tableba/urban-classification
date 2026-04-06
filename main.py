from src.data.dataset import get_dataloader
from src.models.train import train_model

def main():
    train_loader = get_dataloader(split="train")
    val_loader = get_dataloader(split="val")
    train_model(train_loader, val_loader)

if __name__ == "__main__":
    main()
