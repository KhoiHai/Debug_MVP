import os
import torch
from torch.optim import AdamW
from src.models.mvp_seg import MVP_Seg
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.models.loss import Model_Loss

def move_targets_to_device(targets, device):
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]

def build_optimizer(model, base_lr, weight_decay, backbone_lr_ratio=0.5):
    return AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * backbone_lr_ratio},
        {"params": model.neck.parameters(), "lr": base_lr},
        {"params": model.pred_head.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)

def poly_lr_scheduler(optimizer, base_lrs, curr_iter, max_iter, power=1.0):
    factor = (1 - curr_iter / max_iter) ** power
    for i, g in enumerate(optimizer.param_groups):
        g["lr"] = base_lrs[i] * factor

def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = get_sbd_dataloaders(
        root=config["data_root"], batch_size=config["batch_size"], num_workers=config["num_workers"]
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    model = MVP_Seg(
        model_name=config["backbone"],
        num_classes=config["num_classes"],
        num_prototypes=config["num_prototypes"]
    ).to(device)

    # Freeze backbone
    for p in model.backbone.parameters(): p.requires_grad = False
    print(f"Backbone frozen for {config['warmup_epochs']} epochs")

    optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    criterion = Model_Loss(num_classes=config["num_classes"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    total_iters = config["epochs"] * len(train_loader)
    warmup_iters = 1500
    start_epoch, best_loss = 0, float("inf")

    # Resume
    if config.get("resume", False) and os.path.exists(config.get("resume_path", "")):
        print(f"🔄 Resuming from {config['resume_path']}")
        ckpt = torch.load(config["resume_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        scaler.load_state_dict(ckpt.get("scaler_state", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        if start_epoch >= config["warmup_epochs"]:
            for p in model.backbone.parameters(): p.requires_grad = True
            optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
            base_lrs = [g["lr"] for g in optimizer.param_groups]

    # ═════════════════════════════════════════
    # Train loop
    # ═════════════════════════════════════════
    for epoch in range(start_epoch, config["epochs"]):
        if epoch == config["warmup_epochs"]:
            print(f"\nEpoch {epoch+1}: Unfreezing backbone")
            for p in model.backbone.parameters(): p.requires_grad = True
            optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
            base_lrs = [g["lr"] for g in optimizer.param_groups]

        model.train()
        total_loss_cls = 0.0
        total_loss_box = 0.0
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
            images, targets = batch
            images = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss_cls = loss_dict["loss_cls"]
                loss_box = loss_dict.get("loss_box", torch.tensor(0.0, device=device))
                loss = loss_cls + loss_box

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10.0)
            scaler.step(optimizer)
            scaler.update()

            # LR schedule
            global_iter = epoch * len(train_loader) + batch_idx
            if global_iter < warmup_iters:
                factor = global_iter / warmup_iters
                for i, g in enumerate(optimizer.param_groups): g["lr"] = base_lrs[i] * factor
            else:
                poly_lr_scheduler(optimizer, base_lrs, global_iter, total_iters)

            total_loss_cls += loss_cls.item()
            total_loss_box += loss_box.item()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}] Step [{batch_idx}/{len(train_loader)}] "
                      f"Loss_cls: {loss_cls.item():.4f} | Loss_box: {loss_box.item():.4f} | Total_loss: {loss.item():.4f}")

        avg_loss_cls = total_loss_cls / max(len(train_loader), 1)
        avg_loss_box = total_loss_box / max(len(train_loader), 1)
        avg_total_loss = total_loss / max(len(train_loader), 1)
        print(f"\nEpoch {epoch+1} Summary | Avg Loss_cls: {avg_loss_cls:.4f} | Avg Loss_box: {avg_loss_box:.4f} | Avg Total Loss: {avg_total_loss:.4f}")

        # Validation
        model.eval()
        val_loss_cls = 0.0
        val_loss_box = 0.0
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, targets = batch
                images = images.to(device)
                targets = move_targets_to_device(targets, device)
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss_cls = loss_dict["loss_cls"].item()
                loss_box = loss_dict.get("loss_box", 0.0)
                loss_total = loss_cls + loss_box

                val_loss_cls += loss_cls
                val_loss_box += loss_box
                val_total_loss += loss_total

        val_loss_cls /= max(len(val_loader), 1)
        val_loss_box /= max(len(val_loader), 1)
        val_total_loss /= max(len(val_loader), 1)
        print(f"Validation Summary | Loss_cls: {val_loss_cls:.4f} | Loss_box: {val_loss_box:.4f} | Total_loss: {val_total_loss:.4f}")

        # Save checkpoint
        os.makedirs(config["save_dir"], exist_ok=True)
        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_loss": best_loss}
        if (epoch+1) % 10 == 0: torch.save(ckpt, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(ckpt, os.path.join(config["save_dir"], "best.pth"))
            print(f"🔥 Best model saved: {best_loss:.4f}")

if __name__ == "__main__":
    config = {
        "data_root": "/content/SBD",
        "save_dir": "checkpoints",
        "backbone": "nvidia/MambaVision-T-1K",
        "num_classes": 20,
        "num_prototypes": 32,
        "batch_size": 2,
        "num_workers": 2,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "epochs": 10,
        "warmup_epochs": 1,
        "resume": False,
        "resume_path": "checkpoints/best.pth",
    }
    train(config)