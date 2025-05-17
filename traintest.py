import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# All different training schemes that run_training.py accesses are bundled here
# not all of them require all arguments (e.g. epsilon) 
# but this way one can keep a constistent interface

def train_plain(model, device, train_loader, optimizer, epoch, 
                lam=1., verbose=100):
    # lam not necessarily needed but there to ensure that the 
    # learning rates on the base and the CEDA model are comparable
    
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=torch.float)
    

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = p_in*criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
    if verbose>0:
        print(f'Train Epoch: {epoch} [{correct}/{1000} ({correct/10:.0f}%)]\tLoss: {train_loss:.6f}')
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset), 0.


def train_rbf_vol(model, device, train_loader, optimizer, epoch,
                lam=1., verbose=100):
    # lam not necessarily needed but there to ensure that the
    # learning rates on the base and the CEDA model are comparable

    criterion = nn.NLLLoss()
    model.train()

    train_loss = 0
    correct = 0

    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=torch.float)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = p_in * criterion(output, target) + model.rbf.get_shapes.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
    if verbose > 0:
        print(f'Train Epoch: {epoch} [{correct}/{1000} ({correct / 10:.0f}%)]\tLoss: {train_loss:.6f}')
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset), 0.

# CEDA as introduced in https://arxiv.org/pdf/1812.05720.pdf
# uses either uniform noise or noise_loader
# not used in paper because conceptually identical to outlier-exposure https://arxiv.org/abs/1812.04606
def train_CEDA(model, device, train_loader, optimizer, epoch, 
               lam=1., verbose=100, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=torch.float)
    p_out = torch.tensor(lam, device=device, dtype=torch.float) * p_in
    
    if noise_loader is not None:
        enum = enumerate(noise_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)
        
        if noise_loader is not None:
            try:
                noise = enum.__next__()[1][0].to(device)
            except: # added for error when the noise loader is not as long as the train_loader
                enum = enumerate(noise_loader)
                noise = enum.__next__()[1][0].to(device)
                
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        loss1 = criterion(output, target)
        loss2 = - output_adv.mean()

        loss = p_in*loss1 + p_out*loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print(f'Train Epoch: {epoch} [{correct}/{len(target)} ({correct / len(target) * 100:.0f}%)]\tLoss: {train_loss:.6f}')
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset), 0.


# CEDA training for case where one fits a CCU model with p(x|o)=1
# not used in paper because there we fit p(x|o)
def train_CEDA_gmm(model, device, train_loader, optimizer, epoch, 
                   lam=1., verbose=100, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    likelihood_loss = 0
    correct = 0
    
    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=torch.float)
    p_out = torch.tensor(lam, device=device, dtype=torch.float) * p_in
    
    log_p_in = p_in.log()
    log_p_out = p_out.log()
    
    if noise_loader is not None:
        enum2 = enumerate(iter(noise_loader))
    
    enum = enumerate(train_loader)
    for batch_idx, (data, target) in enum:
        data, target = data.to(device), target.to(device)
        

        if noise_loader is not None:
            try:
                noise = enum2.__next__()[1][0].to(device)
            except StopIteration: # added for error when the noise loader is not as long as the train_loader
                # ended up being error with batch size and iterator mechanics
                print("WARNING: OOD_loader and train_loader not the same size... resampling OOD")
                enum2 = enumerate(iter(noise_loader))
                noise = enum2.__next__()[1][0].to(device)
        else:
            noise = torch.rand_like(data)
                


        optimizer.zero_grad()
        
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        like_in = torch.logsumexp( model.mm(data.view(data.shape[0], -1)), 0 )
        like_out =  torch.logsumexp( model.mm(noise.view(noise.shape[0], -1)), 0 )
        
        loss1 = criterion(output, target)
        loss2 = - output_adv.mean()
        loss3 = - torch.logsumexp(torch.stack([log_p_in + like_in, 
                                               log_p_out*torch.ones_like(like_in)], 0), 0).mean()
        loss4 = - torch.logsumexp(torch.stack([log_p_in + like_out, 
                                               log_p_out*torch.ones_like(like_out)], 0), 0).mean()
        
        loss =  p_in*(loss1 + loss3) + p_out*(loss2 + loss4)
        
        loss.backward()
        optimizer.step()
        
        likelihood_loss += loss3.item()
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        if (batch_idx % verbose == 0) and verbose>0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}\
 ({100. * batch_idx / len(train_loader):.0f})%]\tLoss: {loss.item():.6f}\t\
Accuracy: {correct / len(train_loader.dataset):.6f}")
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
             #   epoch, batch_idx * len(data), len(train_loader.dataset),
              #  100. * batch_idx / len(train_loader), loss.item()))
    return (train_loss / len(train_loader.dataset), 
            correct / len(train_loader.dataset), 
            likelihood_loss / len(train_loader.dataset))


# CEDA training for CCU model where both in- and out-distribution are learned
# the margin is introduced to ensure that the sigmas of the out-distribution are
# always larger than for the in-distribution
margin = np.log(4.)

def train_CEDA_gmm_out(model, device, train_loader, optimizer, epoch, 
                   lam=1., verbose=100, noise_loader=None, epsilon=.3,
                       dtype=torch.float):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    likelihood_loss = 0
    correct = 0
    
    p_in = torch.tensor(1. / (1. + lam), device=device, dtype=dtype)
    p_out = torch.tensor(lam, device=device, dtype=dtype) * p_in
    
    log_p_in = p_in.log()
    log_p_out = p_out.log()
    
    if noise_loader is not None:
        enum2 = enumerate(iter(noise_loader))
    
    enum = enumerate(train_loader)
    for batch_idx, (data, target) in enum:
        data, target = data.to(device), target.to(device)
        
        if noise_loader is not None:
            try:
                noise = enum2.__next__()[1][0].to(device)
            except StopIteration: # added for error when the noise loader is not as long as the train_loader
                # ended up being error with batch size and iterator mechanics
                print("WARNING: OOD_loader and train_loader not the same size... resampling OOD")
                enum2 = enumerate(iter(noise_loader))
                noise = enum2.__next__()[1][0].to(device)

        optimizer.zero_grad()
        
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        
        like_in_in = torch.logsumexp( model.mm(data.view(data.shape[0], -1)), 0 )
        like_out_in =  torch.logsumexp( model.mm(noise.view(noise.shape[0], -1)), 0 )
        
        like_in_out = torch.logsumexp( model.mm_out(data.view(data.shape[0], -1)), 0 )
        like_out_out =  torch.logsumexp( model.mm_out(noise.view(noise.shape[0], -1)), 0 )
        
        
        loss1 = criterion(output, target)
        loss2 = - output_adv.mean()
        loss3 = - torch.logsumexp(torch.stack([log_p_in + like_in_in, 
                                               log_p_out + like_in_out], 0), 0).mean()
        loss4 = - torch.logsumexp(torch.stack([log_p_in + like_out_in, 
                                               log_p_out + like_out_out], 0), 0).mean()
        
        
        loss =  p_in*(loss1 + loss3) + p_out*(loss2 + loss4)
        
        loss.backward()
        optimizer.step()
        
        likelihood_loss += loss3.item()
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        
        threshold = model.mm.logvar.max() + margin
        idx = model.mm_out.logvar<threshold
        model.mm_out.logvar.data[idx] = threshold

    if verbose>0:
        print(f'Train Epoch: {epoch} [{correct}/{1000} ({correct / 10:.0f}%)]\tLoss: {train_loss:.6f}')
    return (train_loss / len(train_loader.dataset), 
            correct / len(train_loader.dataset), 
            likelihood_loss / len(train_loader.dataset))



def test(model, device, test_loader, min_conf=.1):
    model.eval()
    test_loss = 0
    correct = 0.
    av_conf = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            c, pred = output.max(1, keepdim=True) # get the index of the max log-probability
            correct += (pred.eq(target.view_as(pred))*(c.exp()>=min_conf)).sum().item()
            av_conf += c.exp().sum().item()
            
    test_loss /= len(test_loader.dataset)
    av_conf /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    
    return correct, av_conf, test_loss





