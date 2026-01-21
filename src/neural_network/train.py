import torch
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import UNet
import torch.nn as nn
import os
from tqdm import tqdm  # Pentru a vedea progresul în timp real

# Setează device-ul pentru Apple Silicon (M1/M2/M3)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"🚀 Antrenarea rulează pe: {DEVICE}")

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4

def main():
    # Creare directoare necesare
    os.makedirs("checkpoints", exist_ok=True)

    train_ds = ChangeDetectionDataset('data/train')
    val_ds   = ChangeDetectionDataset('data/validation')

    # num_workers=0 este recomandat pe Mac pentru a evita erorile de memorie partajată
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, num_workers=0)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Folosim BCEWithLogitsLoss dacă modelul tău NU are Sigmoid la final. 
    # Dacă are deja Sigmoid, rămâi la nn.BCELoss()
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Adăugăm bara de progres
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for x, y in loop:
            # Ne asigurăm că datele sunt pe GPU-ul Mac-ului și sunt tip Float
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()

            optimizer.zero_grad()
            pred = model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Actualizăm informațiile din bara de progres
            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} finalizată. Loss mediu: {avg_loss:.4f}")
        
        # Salvare checkpoint după fiecare epocă (siguranță)
        torch.save(model.state_dict(), f"checkpoints/last_model.pth")

    torch.save(model.state_dict(), "checkpoints/unet_final.pth")
    print("✨ Antrenare completă! Modelul final a fost salvat în folderul 'checkpoints'.")

if __name__ == "__main__":
    main()