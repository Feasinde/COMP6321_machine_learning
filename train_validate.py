import torch
from tqdm import tqdm
def train(input_tuple,model,optimiser,criterion):
    model.train()
    input_tensor=input_tuple[0]
    target=input_tuple[1]
    optimiser.zero_grad()
    prediction = model(input_tensor)
    loss = criterion(prediction,target)
    loss.backward()
    clip_grad_norm_(model.parameters(),clip)
    optimiser.step()
    return loss.item(), prediction

def valid(input_tuple,model,criterion):
    model.eval()
    input_tensor=input_tuple[0]
    target=input_tuple[1]
    prediction = model(input_tensor)
    loss = criterion(prediction,target)
    return loss.item(), prediction

def trainIters(model,
               train_dset,
               valid_dset,
               batch_size,
               n_epochs,
               learning_rate,
               weight_decay):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    train_dl = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)
    train_dl=tqdm(train_dl,desc='Training',leave=False)
    valid_dl = torch.utils.data.DataLoader(valid_dset, batch_size=batch_size)
    valid_dl=tqdm(valid_dl,desc='Validating',leave=False)
    
    for epoch in tqdm(range(1,n_epochs+1),desc='Epoch'):
        for x,y in train_dl:
            input_tensor = x.to(device)
            target = y.to(device)
            train_loss, prediction = train(input_tensor,target,model,optimiser,criterion)
        
        with torch.no_grad():
            valid_loss = 0
            for x, y in valid_dl:
                input_tensor = x.to(device)
                target = y.to(device)
                v_loss, valid_pred = valid(input_tensor,target,model,criterion)