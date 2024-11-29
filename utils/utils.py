import torch

def train_fn(dataloader,model,loss_fn,optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch,(x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        if batch == 0:
            print(x.shape)
        #lossの計算
        pred = model(x)
        loss = loss_fn(pred,y)

        #誤差逆伝播方
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss,current = loss.item(), batch*len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_fn(dataloader,model,loss_fn,device):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batchs
    correct /= size
    print(f"Test Error: \\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \\n")
