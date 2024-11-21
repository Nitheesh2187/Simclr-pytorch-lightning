from torchvision import transforms
from utils import get_color_distortion
from models import LitSimclr
from dataset import CIFAR10PairDataModule
from torchvision.models import resnet18
import lightning as L
import warnings
warnings.filterwarnings("ignore")


learning_rate = 0.6
momentum = 0.9
temperature = 0.5  
num_epochs = 1
model_transforms = {
                    "train":transforms.Compose([transforms.RandomResizedCrop(32),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            get_color_distortion(),
                                            transforms.ToTensor()]),
                    "test":transforms.ToTensor()
}

dm = CIFAR10PairDataModule(
    data_dir="./data",
    batch_size=256,
    transforms=model_transforms,
    num_workers=4
)

train_loader_len = len(dm.train_dataloader())

model = LitSimclr(
        backbone=resnet18,
        lr = learning_rate,
        momentum=momentum,
        temperature=temperature,
        num_epochs=num_epochs,
        train_loader_len=train_loader_len,
        projection_dim=128
    )


trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=num_epochs,
    )



trainer.fit(model,dm)
trainer.validate(model,dm)
trainer.test(model,dm)