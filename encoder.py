import torch.nn as nn

class auto_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_block1 = self.set_block(1,5)
        self.encoder_block2 = self.set_block(5,10)
        self.encoder_block3 = self.set_block(10,20,relu=False)
        self.decoder_block3 = self.set_block(20,10)
        self.decoder_block2 = self.set_block(10,5)
        self.decoder_block1 = self.set_block(5,1,relu=False)

        self.pool = nn.MaxPool2d(2,return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

    def set_block(self,input,output,relu=True):
        self.block = nn.Sequential(nn.Conv2d(input,output,3,padding=1),
                                   nn.BatchNorm2d(output),
                                   nn.ReLU() if relu==True else nn.Tanh()
                                   )
        return self.block

    def forward(self,x):
        x = self.encoder_block1(x) #28*28
        x,index1 = self.pool(x)   #14*14
        x = self.encoder_block2(x) #14*14
        x,index2 = self.pool(x)  #7*7
        latent = self.encoder_block3(x) #7*7

        x = self.decoder_block3(latent) #7*7
        x = self.unpool(x,index2) #14*14
        x = self.decoder_block2(x) #14*14
        x = self.unpool(x,index1) #28*28
        x = self.decoder_block1(x)
        return latent,x