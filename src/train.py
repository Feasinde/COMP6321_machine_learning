import torch,IPython
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(input_tensor,target,model,optimiser,criterion):
    model.train()
    optimiser.zero_grad()
    prediction = model((input_tensor,target)).permute(0,2,1)
    loss = criterion(prediction,target)
    loss.backward()
    # clip_grad_norm_(model.parameters(),clip)
    optimiser.step()
    return loss.item(), prediction

def valid(input_tensor,target,model,criterion):
    model.eval()
    prediction = model((input_tensor,target)).permute(0,2,1)
    loss = criterion(prediction,target)
    return loss.item(), prediction

def trainIters(model,
               train_dset,
               valid_dset,
               batch_size,
               n_epochs,
               learning_rate,
               weight_decay):
    print("CUDA is available!" if torch.cuda.is_available() else "NO CUDA 4 U")

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dl=torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
    
    
    valid_dl=torch.utils.data.DataLoader(valid_dset, batch_size=batch_size)
    
    
    train_losses=[np.inf]
    valid_losses=[np.inf]

    for epoch in tqdm(range(1,n_epochs+1),desc='Epoch',leave=False):
        plt.gca().cla()
        plt.xlim(0,n_epochs)
        plt.ylim(0,1)
        plt.title("Learning curve")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.text(n_epochs/2,.9,"Train loss: {:.2f}".format(train_losses[-1]))
        plt.text(n_epochs/2,.8,"Validation loss: {:.2f}".format(valid_losses[-1]))
        plt.plot(train_losses, "-b", label="Training loss")
        plt.plot(valid_losses, "-r", label="Validation loss")
        plt.legend(loc="upper left")
        IPython.display.display(plt.gcf())
        
        train_dl=tqdm(train_dl,desc='Training',leave=False)
        for x,y in train_dl:
            input_tensor = x.to(device)
            target = y.to(device)
            train_loss, prediction = train(input_tensor,target,model,optimiser,criterion)
#         print("Training loss: {:.4f}".format(train_loss))
        train_losses.append(train_loss)
        
        with torch.no_grad():
            valid_loss = 0
            valid_dl=tqdm(valid_dl,desc='Validating',leave=False)
            for x, y in valid_dl:
                input_tensor = x.to(device)
                target = y.to(device)
                v_loss, valid_pred = valid(input_tensor,target,model,criterion)
#             print("Validation loss: {:.4f}".format(v_loss))
            valid_losses.append(v_loss)
        IPython.display.clear_output(wait=True)
    return train_losses,valid_losses
