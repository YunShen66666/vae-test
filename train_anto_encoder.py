import torch
from torchvision import datasets,transforms
import torch.nn as nn
from encoder import auto_Encoder
import time

BATCH_SIZE = 1000
epoch = 20
learning_rate = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
mnist = datasets.MNIST(root='./',train=True,transform=transforms,download=True)
#print(mnist[0][0])

dataloader = torch.utils.data.DataLoader(mnist,batch_size=BATCH_SIZE,shuffle=True)

auto_E = auto_Encoder().to(DEVICE)
optimizer = torch.optim.SGD(auto_E.parameters(),lr=learning_rate)
citizerion = nn.MSELoss()


for i in range(epoch):
    a = time.time()
    auto_E.train()
    for index,(img,_) in enumerate(dataloader):
        x = img.to(DEVICE)
        latent,output = auto_E(x)
        loss = citizerion(output,x)
        loss.backward()
        optimizer.step()
        b = time.time()
        if index%10==9:
            b = time.time()
            print("epoch={},batch={},loss={:.4f},speed={:.6f}/10个batch".format(i,index,loss,index/(b-a)))
    torch.save(auto_E.state_dict(),'./')



