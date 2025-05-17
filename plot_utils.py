import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import pandas as pd
from functools import partial


def plot_conf(net,
              data,
              labels,
              ax = None,
              device = None,
              title: str = '',
              domain: tuple = (-.03, 1.03),
              grid_size: float = .01,
              ):
    """
    net: takes in classifier network
    data: tensor with input data (expected normalized [0,1])
    labels: true label values for plotting
    ax: axis to plot on if doing a subplot
    device: torch device for training
    """
    x = y = np.arange(domain[0], domain[1], grid_size)
    points = []
    for xx in x:
        for yy in y:
            points.append([xx, yy])
    dim = len(x)
    net.to(device)
    points = torch.tensor(points, dtype=torch.float32, device=device)
    output = net(points).detach()
    output_data = net(data.to(device)).detach().cpu()
    pred = output.max(1)[0].exp()
    z = pred.view(dim, dim).detach().t().cpu().numpy()
    # COLORS FOR CLASSES IN CLASS VS OOD
    p, yhat = output_data.max(1)
    p = p.exp()
    acc = (yhat.eq(labels)).sum() / len(labels)
    if ax: # if axis input plot to that
        cont = ax.contourf(x, y, z, vmin=.5, vmax=1, extend='both', cmap='bone', zorder=0)
        cax = ax.inset_axes([1, 0, 0.04, 1.045])
        cb = plt.colorbar(cont, ax=ax, cax=cax, orientation='vertical')
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=yhat, ax=ax)
        ax.title.set_text(f'{title}\nAccuracy:{acc:.3f}')
    else:

        plt.contourf(x, y, z, vmin=.5, vmax=1, extend='both', cmap='bone', zorder=0)
        plt.colorbar()  # ticks=np.linspace(.5,1.,6))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=yhat)
        plt.title(f'{title}\nAccuracy:{acc:.3f}')

def kernel_map(rbf,
               df,
               ax = None,
               ):
    if ax:
        pass
    else:
        fig, ax = plt.subplots()


    kernels = rbf.get_kernels_centers.cpu().numpy()
    shapes = rbf.get_shapes.cpu().numpy()
    norm = rbf.norm_function
    radial_f = rbf.radial_function
    x2 = np.arange(min(kernels[:,0].min(),0), max(kernels[:,0].max(),1), 0.05)
    y2 = np.arange(min(kernels[:,1].min(),0), max(kernels[:,1].max(),1), 0.05)  # assumes feature space is two-dimensional and in range of kernels
    rep = x2.shape[0] * y2.shape[0]
    X, Y = np.meshgrid(x2, y2)
    for i in range(len(kernels)):
        center = kernels[i][:, None].repeat(rep, axis=1).T
        zs = np.array(radial_f(
            torch.tensor(shapes[i].repeat(rep)) * norm(torch.tensor(center.T - [X.ravel(), Y.ravel()]),
                                                                          dim=0)))
        Z = zs.reshape(X.shape)
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Learnt kernels')
    ax.set_xlabel('feature_1')
    ax.set_ylabel('feature_2')

    sns.scatterplot(x=df.feature_1, y=df.feature_2, hue=df.labels, ax=ax)
    sns.scatterplot(x=kernels[:,0], y=kernels[:,1],palette='Set2', s = 130,
                    hue=np.array(['rbf_center']).repeat(len(kernels)), ax=ax)

def _animate_em(frame,
                zs,
               x,
               y,
               ax,
               data
                ):
    #print(frame)
    # unpack first z and center values
    z = zs[frame][0]
    center = zs[frame][1]
    df2 = pd.DataFrame(center, columns=['feature_1', 'feature_2'])
    df2['labels'] = ['gmm_center'] * len(center)
    # first plot

    ax.cla()
    cont = ax.contourf(x, y, z, cmap='bone')
    sns.scatterplot(data=data, x='feature_1', y='feature_2',
                    hue='labels', ax=ax)
    sns.scatterplot(data = df2, x = 'feature_1', y = 'feature_2',
                    hue = 'labels', palette='Set2', s = 130, ax=ax)
    ax.title.set_text(f'EM algorithm learning\nEM Step: {frame}')

def animate_em(zs,
               x,
               y,
               data):
    #print('start --------------------------------------------------------')
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, partial(_animate_em, zs=zs, x=x,y=y,ax=ax,data=data),
                                  frames=min(int(len(zs)/2), 31), repeat=False)
# To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=2,
                                 metadata=dict(artist='SLeathersII'),
                                 bitrate=1800)
    ani.save('EM.gif', writer=writer)

def _animate_rbf(frame,
                 rbfs,
                 df,
                 ax):
    # iterate rbfs
    rbf = rbfs[frame]
    ax.cla() # clear axis for next drawing
    kernel_map(rbf, df, ax=ax) # draw step
    ax.title.set_text(f'Kernel Learning\nEpoch: {frame}')# replace title
def animate_rbf(rbfs,
                df,
                title=' '):
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, partial(_animate_rbf, rbfs=rbfs, df=df, ax=ax),
                                  frames=len(rbfs), repeat=False)
    writer = animation.PillowWriter(fps=5,
                                    metadata=dict(artist='SLeathersII'),
                                    bitrate=1800)

    ani.save(f'Kernel{title}.gif', writer=writer)
