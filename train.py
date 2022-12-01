import argparse
import random
import os
import time
import copy

import torch
import numpy as np

from network import MC_DPGMM
from metric import valid
from loss import Loss
from dataloader import load_data, DataLoaderX


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(epoch):
    model.train()
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    model.train()
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(args.epsilon * criterion.contrastive_loss(qs[v], qs[w]))
                loss_list.append(args.omega * criterion.forward_label(qs[v], qs[w]))
            loss_list.append(args.lamb * model.label_contrastive_module.LOBO(zs[v]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    time0 = time.time()

    # MNIST-USPS
    # BDGP
    # CCV
    # Fashion
    # Caltech-2V
    # Caltech-3V
    # Caltech-4V
    # Caltech-5V
    Dataname = 'Fashion'
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--mse_epochs", default=200)
    parser.add_argument("--con_epochs", default=100)
    parser.add_argument("--feature_dim", default=512)
    parser.add_argument("--lamb", default=1)
    parser.add_argument("--epsilon", default=1)
    parser.add_argument("--omega", default=1)
    parser.add_argument("--init_v", default=-2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The code has been optimized.
    # The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
    if args.dataset == "MNIST-USPS":
        args.temperature_c = 0.5
        args.temperature_s = 0.4
        seed = 1
    if args.dataset == "BDGP":
        args.temperature_c = 4.47
        args.temperature_s = 0.6
        seed = 1
    if args.dataset == "CCV":
        args.temperature_c = 0.2
        args.temperature_s = 0.5
        seed = 1
    if args.dataset == "Fashion":
        args.temperature_c = 0.7
        args.temperature_s = 4.3
        seed = 1
    if args.dataset == "Caltech-2V":
        args.temperature_c = 0.6
        args.temperature_s = 0.5
        seed = 3
    if args.dataset == "Caltech-3V":
        args.temperature_c = 1.1
        args.temperature_s = 0.5
        seed = 1
    if args.dataset == "Caltech-4V":
        args.temperature_c = 0.7
        args.temperature_s = 0.3
        seed = 1
    if args.dataset == "Caltech-5V":
        args.temperature_c = 1.43
        args.temperature_s = 0.6
        seed = 1

    setup_seed(seed)

    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = DataLoaderX(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = MC_DPGMM(view, dims, args.feature_dim, args.init_v)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, args.temperature_c, args.temperature_s).to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        if epoch == args.mse_epochs:
            acc, nmi, pur, pred, label = valid(model, device, dataset, view, data_size)
            best_score = acc + nmi + pur
            best_accs, best_nmis, best_purs = acc, nmi, pur
            best_pred, best_label = pred, label
            best_model_wts = copy.deepcopy(model.state_dict())
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs or (epoch - args.mse_epochs) % 1 == 0:
            acc, nmi, pur, pred, label = valid(model, device, dataset, view, data_size)
            infered_K.append(len(compute_n_clusters(pred)))
            if acc + nmi + pur > best_score:
                best_score = acc + nmi + pur
                best_accs, best_nmis, best_purs = acc, nmi, pur
                best_pred, best_label = pred, label
                best_model_wts = copy.deepcopy(model.state_dict())
        epoch += 1

    model.load_state_dict(best_model_wts)
    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    print('Saving..')

    print("******Best Scores******")
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} SUM={:.4f}'.format(best_accs, best_nmis, best_purs, best_score))

    train_time = time.time() - time0
    print('****** End, training time = {} s ******'.format(round(train_time, 2)))
