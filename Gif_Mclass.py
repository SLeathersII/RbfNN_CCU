import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn import  datasets
import traintest as tt
import models





class DxW_MClassNet(nn.Module):
    """
        d is number of layers
        w is width of hidden layers
    """

    def __init__(self, d: int = 3,
                 w: int = 32,
                 n_class: int = 3,
                 activation=F.relu,
                 dtype=torch.float):
        """
            d is number of layers
            w is width of hidden layers
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, w, dtype=dtype))
        self.n_class = n_class
        assert d >= 2, 'MLP depth must be at least 3 (2 will break but work for examples)'
        for depth in range(d - 2):
            self.layers.append(nn.Linear(w, w, dtype=dtype))
        self.layers.append(nn.Linear(w, n_class, dtype=dtype))
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer.out_features != self.n_class:  # use the set activation for all but output layer
                x = self.activation(x)
            else:
                y = F.log_softmax(x, dim=1)  # needed for CCU

        return y

# Create grid for contour
x = np.arange(0., 1.01, 0.01)
y = np.arange(0., 1.01, 0.01)

points = []
for xx in x:
    for yy in y:
        points.append((xx, yy))

def plot_conf(net, data, axM, device, labels, modelstr, legend, points=points):
    # plt.clf()

    colors = np.array(['#377eb8', '#ff7f00', '#f781bf',
                       '#4daf4a','#a65628', '#984ea3',
                       '#999999', '#e41a1c', '#dede00'])
    output = net(torch.tensor(points, dtype=torch.float32, device=device)).detach()
    output_data = net(data.to(device)).detach().cpu()
    pred = output.max(1)[0].exp()
    z = pred.view(len(x), len(y)).detach().t().cpu().numpy()
    # COLORS FOR CLASSES IN CLASS VS OOD
    p, yhat = output_data.max(1)
    p = p.exp()
    # alias yhat based on p values ( 2 is ood)
    yhat[p < 0.62] = 3
    acc = (yhat.eq(labels)).sum()/len(labels)
    axM.cla()

    ticks = np.arange(.3, 1.01,.125)
    cont = axM.contourf(x, y, z, ticks, vmin=.5, vmax=1, extend='both', cmap='bone')
    cax = axM.inset_axes([1, 0, 0.04, 1])
    cb = plt.colorbar(cont, ax=axM, cax=cax, orientation='vertical')
    if legend:

        cb.set_label("Confidence", labelpad=0)
    axM.scatter(data[:, 0][yhat==0].cpu(), data[:, 1][yhat.cpu()==0].cpu(), s=3,
                color=colors[yhat[yhat==0]], label='Class 0')
    axM.scatter(data[:, 0][yhat == 1].cpu(), data[:, 1][yhat.cpu() == 1].cpu(), s=3,
                color=colors[yhat[yhat==1]], label="Class 1")
    axM.scatter(data[:, 0][yhat == 2].cpu(), data[:, 1][yhat.cpu() == 2].cpu(), s=3,
                color=colors[yhat[yhat==2]], label="Class 2")
    axM.scatter(data[:, 0][yhat == 3].cpu(), data[:, 1][yhat.cpu() == 3].cpu(), s=3,
                color=colors[yhat[yhat == 3]], label="OOD")
    if legend:
        axM.legend(markerscale=4, bbox_to_anchor=(1.3,1.24), ncols=2 )
    axM.set_xlim(0, 1)
    axM.set_ylim(0, 1)
    #plt.xticks([])
    #plt.yticks([])
    axM.tick_params(left = False, bottom = False,
                    labelleft = False, labelbottom = False)
    axM.set_aspect('equal', adjustable='box')
    axM.title.set_text(f'{modelstr}\nAccuracy:{acc:.3f}')
# train
def train_plain(model, device, train_loader, optimizer, epoch,
                lam=1., verbose=100, noise_loader=None, epsilon=.3):
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

        loss = p_in * criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset), 0.

# training details
device = 'cuda'
lr = 0.1
lr_gmm = 1e-3
decay = 5e-4
##############3 MULTI CLASS ###################################
mc_data, mc_labels = datasets.make_classification(n_samples=1000,
                                                 n_features=2,
                                                 n_informative=2,
                                                 n_redundant=0,
                                                 n_classes=4,
                                                 n_clusters_per_class=1,
                                                 class_sep=1.8)
# in class data
mc_data = torch.tensor((mc_data + 4) / 10).float()
mc_label = torch.tensor(mc_labels, dtype=torch.float)
#label for in/out
mc_io = np.zeros(len(mc_labels))
mc_io[mc_labels==3] = 1
# take last class as OOD
data_out = mc_data[mc_labels==3].float()
data = mc_data[mc_labels!=3].float()
mc_labels_in = mc_labels[mc_labels!=3]
mc_labels_out = mc_labels[mc_labels == 3]

label = torch.tensor(mc_labels_in, dtype=torch.float)
label = label.type(torch.LongTensor)
label_out = torch.tensor(mc_labels_out).type(torch.LongTensor)
# Data loaders
train = data_utils.TensorDataset(data, label)
train_loader = data_utils.DataLoader(train, batch_size=250, shuffle=True)

train_out = data_utils.TensorDataset(data_out, label_out)
train_loader_out = data_utils.DataLoader(train_out, batch_size=250, shuffle=True)

colors = np.array(['#377eb8', '#ff7f00', '#f781bf',
                       '#4daf4a','#a65628', '#984ea3',
                       '#999999', '#e41a1c', '#dede00'])
# show data in class vs out of class and save png
for i in range(4):
    pass
    if i ==3:
        plt.scatter(mc_data[:, 0][mc_label == i], mc_data[:, 1][mc_label == i],
                    label=f"OOD", color=colors[(mc_label[mc_label==i]).int()])
    else:
        plt.scatter(mc_data[:, 0][mc_label == i], mc_data[:, 1][mc_label == i],
                    label=f"Class {i}", color=colors[(mc_label[mc_label==i]).int()])
plt.legend()
plt.title("In Class Vs. Out of Class\nTrue Labels")
plt.show(block=False)
input("Press enter If you'd like to continue...")
plt.savefig("scatter_plot.png")
plt.close()



# f, ax1 = plt.subplots(1, 1)
#
# def MC_animation(epoch: int = 0, net=net,
#             train_loader=train_loader,
#             optimizer=optimizer,
#             data=mc_data, ax1=ax1,
#             device=device):
#
#     _, acc, _ = train_plain(net.to(device), device, train_loader, optimizer,
#                             epoch, verbose=-1)
#
#     plt.suptitle(f'Training Epoch: {epoch}\nAccuracy: {acc:.3f}')
#     plot_conf(net, data, ax1, device)
# print("starting MC.gif")
# ani = animation.FuncAnimation(f, MC_animation, frames=250, repeat=False)
#
# writer = animation.PillowWriter(fps=5,
#                                  metadata=dict(artist='Me'),
#                                  bitrate=1800)
# ani.save('MC.gif', writer=writer)
# print('done')
## base network
net = DxW_MClassNet(d=3, w=32, n_class=3)
param_groups = [{'params':net.parameters(),'lr':lr, 'weight_decay':decay}]
optimizer = optim.Adam(param_groups)
# learn GMMs for CCU
cde_in = CDE.train_gmm(train_loader,
                       centroids = 50,
                       max_train=1000,
                       pca = False,
                       train_mu=True)

cde_out = CDE.train_gmm(train_loader_out,
                       centroids = 50,
                       max_train=1000,
                       pca = False,
                       train_mu=False)
# Initialize CCU
DA = CCU(DxW_MClassNet(d=3),
         cde_in, cde_out,
         n_classes=3,
         device=device,
         data_parallel=False,
         lam=1)
# f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
#f2, ax1 = plt.subplots(1, 1)
# pull out model
ccu = DA.CCU
param_groups_ccu = [{'params':ccu.base_model.parameters(),'lr':lr, 'weight_decay':decay},
                {'params':ccu.density_estimator_in.parameters(), 'lr':lr_gmm, 'weight_decay':0},
                {'params':ccu.density_estimator_out.parameters(), 'lr':lr_gmm, 'weight_decay':0},] # ignore GMMs
optimizer_ccu = optim.Adam(param_groups_ccu)
# def animate_CCU(epoch=1,
#                 net=net,
#                 train_loader=train_loader,
#                 optimizer=optimizer,
#                 data=mc_data,
#                 ax1=ax1,
#                 ax2=ax2,
#                 device=device):
#
#     _, loss_acc = tt.train_ccu(net,
#                                train_loader=train_loader,
#                                ood_loader=train_loader_out,
#                                optimizer=optimizer,
#                                verbose=-1,
#                                device=device,
#                                margin=np.log(25),
#                                 )
#     plt.suptitle(f'Training Epoch: {epoch}\nAccuracy: {loss_acc[1]:.3f}')
#     plot_conf(net, data, ax1, device)
#     plot_conf(net.base_model, data, ax2, device)
#
# print('starting MC_CCU.gif')
# ani = animation.FuncAnimation(f2, animate_CCU, frames=5000, repeat=False)
# #plt.show()
# # To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=5,
#                                  metadata=dict(artist='Me'),
#                                  bitrate=1800)
# ani.save('MC_CCU.gif', writer=writer)

f3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
#plt.tight_layout()
plt.subplots_adjust(bottom=0, top=.82,left=0,wspace=0,right=.95)
def animate_all(epoch=1,
                net=net,
                ccu=ccu,
                train_loader=train_loader,
                optimizer=optimizer,
                optimizer_ccu=optimizer_ccu,
                data=mc_data,
                ax1=ax1,
                ax2=ax2,
                ax3=ax3,
                label=mc_label,
                device=device):
    # train CCU
    _, loss_acc = tt.train_ccu(ccu,
                               train_loader=train_loader,
                               ood_loader=train_loader_out,
                               optimizer=optimizer_ccu,
                               verbose=-1,
                               device=device,
                               margin=np.log(15),
                                )
    # train BASE MODEL
    _, acc, _ = train_plain(net.to(device), device, train_loader, optimizer,
                            epoch, verbose=-1)
    plt.suptitle(f'Training Epoch: {epoch}')
    plot_conf(net, data, ax1, device, label, 'Base Classifier', False)
    plot_conf(ccu.base_model, data, ax2, device, label,
              'Base Classifier Conditioned', False)
    plot_conf(ccu, data, ax3, device, label, 'CCU', True)
    # ax1.title.set_text(f'Base Classifier\nAccuracy:{acc:.3f}')
    # ax2.title.set_text(f'Base Classifier Conditioned\nAccuracy:{loss_acc[1]:.3f}')
    # ax3.title.set_text(f'CCU\nAccuracy:{loss_acc[1]:.3f}')
    #plt.tight_layout()
print('starting MC_ALL.gif')
ani = animation.FuncAnimation(f3, animate_all, frames=500, repeat=False)
#plt.show()
# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=5,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save('MC_ALL.gif', writer=writer)