import torch
from data.dataloader import data_loader_train
from models.network import Networks
import models.metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def graph_loss(Z, S):
    S = 0.5 * (S.permute(1, 0) + S)
    D = torch.diag(torch.sum(S, 1))
    L = D - S
    return 2 * torch.trace(torch.matmul(torch.matmul(Z.permute(1, 0), L), Z))


data_0 = sio.loadmat('rand/fm_2000_edge_ori.mat')
data_dict = dict(data_0)
data0 = data_dict['groundtruth'].T
label_true = np.zeros(2000)
for i in range(2000):
    label_true[i] = data0[i]
# print(label_true)

reg2 = 1.0 * 10 ** (10 / 10.0 - 3.0)

model = Networks()
# model.load_state_dict(torch.load('./models/AE1.pth'))
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
n_epochs = 1001
for epoch in range(n_epochs):
    for data in data_loader_train:
        # train_imga, train_imgb, train_imgc = data
        train_imga, train_imgb = data
        input1 = train_imga.view(2000, 1, 28, 28)
        input2 = train_imgb.view(2000, 1, 28, 28)
        output1, output2 = model(input1, input2)
        # loss = criterion(output, input1, input2)
        loss = 0.5 * criterion(output1, input1) + 0.5 * criterion(output2, input2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("Loss is:{:.4f}".format(loss.item()))
torch.save(model.state_dict(), './models/AE1.pth')

print("step2")
print("---------------------------------------")
criterion2 = torch.nn.MSELoss(reduction='sum')
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
n_epochs2 = 2001
ACC_FM = np.zeros((1, 21))
NMI_FM = np.zeros((1, 21))
for epoch in range(n_epochs2):
    for data in data_loader_train:
        # train_imga, train_imgb, train_imgc = data
        train_imga, train_imgb = data
        input1 = train_imga.view(2000, 1, 28, 28)
        input2 = train_imgb.view(2000, 1, 28, 28)
        z11, z22, z1, zcoef1, output1, coef, z2, zcoef2, output2 = model.forward2(input1, input2)
        loss_re = criterion2(coef, torch.zeros(2000, 2000, requires_grad=True))
        # loss_e = criterion2(zcoef2, z2)
        # loss_r = criterion2(output1, input1)
        loss_e = 0.4 * criterion2(zcoef2, z2) + 0.1 * criterion2(zcoef1, z1)
        loss_r = 0.2 * criterion2(output1, input1) + 0.1 * criterion2(output2, input2)
        loss_g1 = graph_loss(z1, coef)
        loss_g2 = graph_loss(z2, coef)
        zz1 = z1.cpu().detach().numpy()
        zz2 = z2.cpu().detach().numpy()
        l1 = np.sum(np.multiply(zz1, zz2))
        loss = 0.01 * loss_r + 0.1 * loss_re + 0.1 * loss_e + 0.5 * loss_g1/2000 + 0.5 * loss_g2/2000 + 0.001 * l1
        # loss = loss_r + 0.1 * loss_re + reg2 * loss_e
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    if epoch % 100 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs2))
        print("Loss is:{:.4f}".format(loss.item()))
        print("Losse is:{:.4f}".format(loss_e.item()))
        print("Lossr is:{:.4f}".format(loss_r.item()))
        print("Lossg1 is:{:.4f}".format(loss_g1.item()))
        print("Lossg2 is:{:.4f}".format(loss_g2.item()))
        print("L1 is:{:.4f}".format(l1.item()))

        coef = model.weight - torch.diag(torch.diag(model.weight))
        commonZ = coef.cpu().detach().numpy()
        alpha = max(0.4 - (10 - 1) / 10 * 0.1, 0.1)
        commonZ = thrC(commonZ, alpha)
        preds, _ = post_proC(commonZ, 10)
        acc = metrics.acc(label_true, preds)
        nmi = metrics.nmi(label_true, preds)
        ACC_FM[0, int(epoch / 100)] = acc
        NMI_FM[0, int(epoch / 100)] = nmi

        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
              % (acc, nmi))
        if acc >= 0.60:
            Z_path = 'commonZ' + str(epoch)
            sio.savemat(Z_path + '.mat', {'Z': commonZ})
torch.save(model.state_dict(), './models/AE2.pth')
sio.savemat('NMI_FM' + '.mat', {'NMI_FM': NMI_FM})
sio.savemat('ACC_FM' + '.mat', {'ACC_FM': ACC_FM})




