import os
import torch
from torch.optim import AdamW
from src.models.mvp_seg import MVP_Seg
from src.dataset.sbd_dataset import get_sbd_dataloaders
from src.models.loss import Model_Loss

# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = get_sbd_dataloaders(
        root=config["data_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    model = MVP_Seg(
        model_name=config["backbone"],
        num_classes=config["num_classes"],
        num_prototypes=config["num_prototypes"]
    ).to(device)

    # Freeze backbone (warmup)
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    criterion = Model_Loss(num_classes=config["num_classes"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_iters = config["epochs"] * len(train_loader)
    warmup_iters = min(1000, len(train_loader) * 2)

    best_loss = float("inf")

    # ─────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────
    for epoch in range(config["epochs"]):

        # Unfreeze backbone
        if epoch == config["warmup_epochs"]:
            print(f"\n🔥 Unfreezing backbone at epoch {epoch+1}")
            for p in model.backbone.parameters():
                p.requires_grad = True

            optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
            base_lrs = [g["lr"] for g in optimizer.param_groups]

        model.train()

        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_box = 0.0
        total_loss_obj = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, targets = batch
            images = images.to(device)
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)

                loss_cls = loss_dict["loss_cls"]
                loss_box = loss_dict["loss_box"]
                loss_obj = loss_dict["loss_obj"]

                # 🔥 YOLO-style loss scaling
                loss = (
                    1.0 * loss_box +
                    1.0 * loss_obj +
                    0.5 * loss_cls
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 10.0
            )
            scaler.step(optimizer)
            scaler.update()

            # LR schedule
            global_iter = epoch * len(train_loader) + batch_idx

            if global_iter < warmup_iters:
                factor = global_iter / warmup_iters
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = base_lrs[i] * factor
            else:
                poly_lr_scheduler(optimizer, base_lrs, global_iter, total_iters)

            # Logging
            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_box += loss_box.item()
            total_loss_obj += loss_obj.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"cls: {loss_cls.item():.4f} | "
                    f"obj: {loss_obj.item():.4f} | "
                    f"box: {loss_box.item():.4f} | "
                    f"total: {loss.item():.4f}"
                )

            # Debug (rất quan trọng)
            if batch_idx % 100 == 0:
                obj_pred = outputs["obj"][0].detach().sigmoid()
                print("   🔍 obj mean:", obj_pred.mean().item())

        # Epoch summary
        num_batches = max(len(train_loader), 1)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  cls: {total_loss_cls / num_batches:.4f}")
        print(f"  obj: {total_loss_obj / num_batches:.4f}")
        print(f"  box: {total_loss_box / num_batches:.4f}")
        print(f"  total: {total_loss / num_batches:.4f}")

        # ─────────────────────────────────────────
        # Validation
        # ─────────────────────────────────────────
        model.eval()

        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                images, targets = batch
                images = images.to(device)
                targets = move_targets_to_device(targets, device)

                outputs = model(images)
                loss_dict = criterion(outputs, targets)

                loss_cls = loss_dict["loss_cls"].item()
                loss_box = loss_dict["loss_box"].item()
                loss_obj = loss_dict["loss_obj"].item()

                loss = (
                    1.0 * loss_box +
                    0.5 * loss_obj +
                    0.5 * loss_cls
                )

                val_total += loss

        val_total /= max(len(val_loader), 1)

        print(f"\n📊 Validation Loss: {val_total:.4f}")

        # Save
        os.makedirs(config["save_dir"], exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss
        }

        if val_total < best_loss:
            best_loss = val_total
            torch.save(ckpt, os.path.join(config["save_dir"], "best.pth"))
            print(f"🔥 Saved best model: {best_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(config["save_dir"], f"epoch_{epoch+1}.pth"))


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "data_root": "/content/SBD",
        "save_dir": "checkpoints",
        "backbone": "nvidia/MambaVision-T-1K",
        "num_classes": 20,
        "num_prototypes": 32,
        "batch_size": 2,
        "num_workers": 2,

        "lr": 1e-4,

        "weight_decay": 0.01,
        "epochs": 10,
        "warmup_epochs": 1,
    }

    train(config)