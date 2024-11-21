from typing import Any
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils import get_lr,nt_xent
import lightning as L
import torch.nn.functional as F

class LitSimclr(L.LightningModule):
    def __init__(self, backbone, lr,momentum,temperature,num_epochs,train_loader_len,projection_dim=128):
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.train_loader_len = train_loader_len
        self.encoder = backbone(weights=False)  

        self.encoder.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = torch.nn.Identity()

        self.feature_dim = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        self.save_hyperparameters()

        # Projection head
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)

        projection = self.projector(features)

        return features, projection

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch,batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss",loss,prog_bar=True,on_step=True,on_epoch=True,logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch,batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("Val_loss",loss,prog_bar=True,on_step=True,on_epoch=True,logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch,batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("Test_loss",loss,prog_bar=True,on_step=True,on_epoch=True,logger=True)
        return loss
    
    def _common_step(self,batch,batch_idx):
        x,_= batch
        # Reshape to combine the images into a single batch
        sizes = x.size()
        x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)
        features = self.encoder(x)
        projections = self.projector(features)
        loss = nt_xent(projections, self.temperature)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
                                    self.parameters(),
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=1.0e-6,
                                    nesterov=True)
        scheduler = LambdaLR(
                            optimizer,
                            lr_lambda=lambda step: get_lr(  
                                step,
                                self.num_epochs * self.train_loader_len,
                                self.lr,
                                1e-3))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

class LinearClassifierLightening(L.LightningModule):
    def __init__(self, encoder: torch.nn.Module, lr,momentum,num_epochs,train_loader_len,input_size, num_classes) -> None:
        super(LinearClassifierLightening, self).__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(input_size, num_classes)
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.train_loader_len = train_loader_len
        self.save_hyperparameters(ignore=['encoder'])
        
    def forward(self, x):
        output = self.fc(self.encoder(x))
        return output    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)
        self.log("train_loss",loss,prog_bar=True,on_step=True,on_epoch=True)
    
    def validation_step(self, batch,batch_idx):
        x,y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)
        self.log("val_loss", loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)
        self.log("test_loss", loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def predict_step(self,batch,batch_idx) :
        x,y = batch
        scores = self(x)
        preds = torch.argmax(scores,dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
                                    self.parameters(),
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=1.0e-6,
                                    nesterov=True)
        scheduler = LambdaLR(
                            optimizer,
                            lr_lambda=lambda step: get_lr(  
                                step,
                                self.num_epochs * self.train_loader_len,
                                self.lr,
                                1e-3))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}