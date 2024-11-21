from dataset import CIFAR10DataModule
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from models import LinearClassifierLightening,LitSimclr
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils import check_accuracy

n_classes = 10
samples_per_class = 250

learning_rate = 0.6
momentum = 0.9
temperature = 0.5  
num_epochs = 3

eval_model_transforms  = {
                    "train":transforms.Compose([transforms.RandomResizedCrop(32),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor()]),
                    "test":transforms.ToTensor()
}

eval_dm = CIFAR10DataModule(
    data_dir="./Data",
    batch_size=256,
    transforms= eval_model_transforms,
    num_workers=4
    )
train_loader_len = len(eval_dm.train_dataloader())


encoder = LitSimclr.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=0-step=136.ckpt").encoder



eval_model = LinearClassifierLightening(
    encoder=encoder,
    lr=0.2,
    momentum=0.9,
    input_size=512,
    num_classes=10,
    num_epochs=2,
    train_loader_len=train_loader_len
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=15, verbose=False, mode="min")
model_checkpoint_callback = ModelCheckpoint("Model_checkpoints/",filename="model-{epoch:02d}-{val_loss:.2f}",save_top_k=3,
                                            monitor="val_loss",mode="min")

trainer = L.Trainer(accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=num_epochs,
        callbacks=[early_stop_callback,model_checkpoint_callback])
trainer.fit(eval_model,eval_dm)
trainer.validate(eval_model,eval_dm)
trainer.test(eval_model,eval_dm)


print(f"Accuracy on training set: {check_accuracy(eval_dm.train_dataloader(), eval_model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(eval_dm.val_dataloader(), eval_model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(eval_dm.test_dataloader(), eval_model)*100:.2f}")




