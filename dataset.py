from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning as L
from PIL import Image
from torch.utils.data import random_split

# https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/README.md
class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target

class CIFAR10PairDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size,transforms,num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.transforms = transforms
        self.num_workers = num_workers
        self.prepare_data()
        

    def prepare_data(self):
        CIFAR10Pair(root=self.data_dir, train=True, download=True)
        CIFAR10Pair(root=self.data_dir,train=False,download=True)

    def setup(self, stage):
        entire_ds = CIFAR10Pair(root=self.data_dir, train=True, transform=self.transforms["train"],download=False)
        train_size = int(0.7 * len(entire_ds))
        valid_size = int(len(entire_ds) - train_size)
        train_ds,valid_ds = random_split(entire_ds,[train_size,valid_size])
        if stage == "fit":
            self.train_ds = train_ds
        if stage == 'valid':
            self.valid_ds = valid_ds 
        if stage == 'test':
            self.test_ds = CIFAR10Pair(root=self.data_dir,train=False,transform=self.transforms["test"],download=False)        

    def train_dataloader(self):
        if self.train_ds is None:
            self.setup('fit') 
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=self.num_workers)
    def val_dataloader(self):
        if self.valid_ds is None:
            self.setup("valid")
        return DataLoader(self.valid_ds,batch_size=self.batch_size,shuffle=False,drop_last=True,num_workers=self.num_workers)
    def test_dataloader(self):
        if self.test_ds is None:
            self.setup("test")    
        return DataLoader(self.test_ds,batch_size=self.batch_size,shuffle=False,drop_last=True,num_workers=self.num_workers)

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size,transforms,samples_per_class=250,num_classes=10,num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.transforms = transforms
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.prepare_data()

    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir,train=False,download=True)

    def setup(self, stage):

        test_dataset = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.transforms["test"],
                download=False,
            )
        test_size = int(0.6 * len(test_dataset))
        valid_size = int(len(test_dataset) - test_size)

        test_ds,valid_ds = random_split(test_dataset,[test_size,valid_size]) 
        if stage == "fit":
            self.train_ds = CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.transforms["train"],
                download=False,
            )
        if stage == "valid":
            self.valid_ds = valid_ds
        if stage == "test":
            self.test_ds =   test_ds        

    def train_dataloader(self):
        if self.train_ds is None:
            self.setup('fit') 
        # indices = np.hstack([np.random.choice(np.where(np.array(self.train_ds.targets) == i)[0], self.samples_per_class, replace=False) for i in range(self.num_classes)])
        # sampler = SubsetRandomSampler(indices)
        return DataLoader(self.train_ds,batch_size = self.batch_size,num_workers=self.num_workers)
    
    def val_dataloader(self):
        if self.valid_ds is None:
            self.setup("valid")
        return DataLoader(self.valid_ds,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
    
    def test_dataloader(self):
        if self.test_ds is None:
            self.setup("test")
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)
