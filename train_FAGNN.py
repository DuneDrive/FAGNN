import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# local imports
from imports.ABIDEDataset_cont import ABIDEDataset
from net.FAGNN_age_comp5 import FAGNN
from imports.utils import train_val_test_split
#from imports.important_node_finder_genotype import node_finder

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# Function to initialize the parser
def initialize_parser():
    parser = argparse.ArgumentParser(description='Run FAGNN for Training')
    # Add arguments to parser
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/data/hsm/FAGNN/quadrant_attention/data/age_pred_behavior_multi_days', help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=0, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
    parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.25, help='scheduler shrinking rate')
    parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization')
    parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=0.1, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=0.1, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
    parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
    parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
    parser.add_argument('--lamb6', type=float, default=0, help='multi-head symmetry regularization')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--ratio', type=float, default=0.3, help='pooling ratio')
    parser.add_argument('--indim', type=int, default=332, help='feature dim')
    parser.add_argument('--nroi', type=int, default=332, help='num of ROIs')
    parser.add_argument('--nclass', type=int, default=1, help='num of classes')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')

    return parser.parse_args()


# Loss functions


def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res


def symmetric_loss(x):
    batch_size = x.size(0) // opt.indim
    quad_dim = int(opt.indim/2)
    x = x.view(batch_size, opt.indim, opt.indim)
    x = (x - torch.mean(x, dim=(0), keepdim=True)) / (torch.std(x, dim=(0), keepdim=True) + 1e-10)
    q1 = x[:, :quad_dim, :quad_dim]
    q2 = x[:, :quad_dim, quad_dim:]
    q3 = x[:, quad_dim:, quad_dim:]
    q4 = x[:, quad_dim:, :quad_dim]
    mse1 = F.mse_loss(q1, q3)
    mse2 = F.mse_loss(q2, q4)
    return mse1 + mse2


# dataloader for train and validation dataset
def prepare_dataloaders(dataset, tr_index_arr, te_index_arr, opt):
    tr_index = tr_index_arr.tolist()
    val_index = te_index_arr.tolist()
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
    return train_loader, val_loader


def train_model(model, train_loader, optimizer, scheduler, device, opt, epoch, writer):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s0_list = []
    s1_list = []
    s2_list = []
    s0_2_list = []
    cnn_att_list = []
    s_edge_list = []
    loss_all = 0
    step = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, w1, w2, s0, s1, s2, s0_2, cnn_att, s_edge = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos, data.pheno, data.behav)
        s0_list.append(s0.view(-1).detach().cpu().numpy())
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())
        s0_2_list.append(s0_2.view(-1).detach().cpu().numpy())
        #cnn_att_list.append(cnn_att.view(-1).detach().cpu().numpy())
        s_edge_list.append(s_edge.detach().cpu().numpy())
        loss_c = F.mse_loss(torch.squeeze(output), torch.squeeze(data.y))
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)

        # Uncomment to use symmetric loss for left and right hemisphere
        #loss_sym = symmetric_loss(s_edge)

        loss_consist = 0

        # Uncomment to use conist loss
        #for c in range(opt.nclass):
        #    loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist #\
                   # Uncomment to use symmetric loss
                   #+ opt.lamb6*loss_sym
        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward(retain_graph=True)
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s0_arr = np.hstack(s0_list)
        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)

    return loss_all / len(train_dataset), s0_list, s1_arr, s2_arr ,w1,w2,s0_2_list,cnn_att_list, s_edge_list


def evaluate_acc(model, loader, device, opt, writer, epoch):
    model.eval()
    correct = 0
    y_pred_arr = np.array([])
    y_arr = np.array([])
    for data in loader:
        data = data.to(device)
        output, w1, w2, s0, s1, s2, s0_2, cnn_att, s_edge = model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos,data.pheno,data.behav)
        y_pred_arr = np.concatenate([y_pred_arr,output.view(-1).detach().cpu().numpy()])
        y_arr = np.concatenate([y_arr,data.y.view(-1).detach().cpu().numpy()])
        acc += abs(torch.sub(torch.squeeze(output),torch.squeeze(data.y))).sum().item()
 
    mae = acc / len(loader.dataset)
    rmse = np.sqrt(np.mean((y_pred_arr - y_arr) ** 2))
    mad_mean = np.mean(np.abs(y_arr - np.mean(y_arr)))
    mad_median = np.mean(np.abs(y_arr - np.median(y_arr)))
    r_squared = 1-(np.sum((y_arr-y_pred_arr)**2))/(np.sum((y_arr-np.mean(y_arr))**2))
    gamma = 1 - (mae/abs(mad_median))

    return mae, rmse, mad_mean, mad_median, r_squared, gamma, y_pred_arr, y_arr


def test_loss(model, loader, device, opt, writer, epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s0, s1, s2, s0_2, cnn_att, s_edge = model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos,data.pheno,data.behav)
        loss_c = F.mse_loss(torch.squeeze(output), torch.squeeze(data.y))
        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)

        # uncomment to use symmetric loss
        #loss_sym = symmetric_loss(s_edge)

        loss_consist = 0

        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 #+ opt.lamb5* loss_consist #\
                  # + opt.lamb6 * loss_sym
                  # Uncomment above to use consist loss and symmetric loss

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)




def run_fagnn(num_sub, num_folds, current_fold, tr_index_arr, te_index_arr, name):
    # Initialize dataset and model

    torch.manual_seed(123)

    EPS = 1e-10
    device = torch.device("cuda:3")

    opt = initialize_parser()
    path = opt.dataroot
    save_model = opt.save_model
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold
    writer = SummaryWriter(os.path.join('./log',str(fold)))
    dataset = ABIDEDataset(path,name)
    attrs = vars(dataset)
    dataset.data.y = dataset.data.y.squeeze()
    history_train = np.zeros([4, num_epoch+1])
    model = FAGNN(opt.indim,opt.ratio,opt.nclass,opt.batchSize).to(device)

    train_loader, val_loader = prepare_dataloaders(dataset, tr_index_arr, te_index_arr, opt):


    # Initialize optimizer and scheduler
    params = [{'params': model.gnn.parameters()}, {'params': model.cnn_1.parameters(), 'lr': 0.002}, {'params': model.cnn_2.parameters(), 'lr': 0.002}, {'params': model.gt.parameters(), 'lr': 0.001}, {'params': model.last_layer.parameters(), 'lr': 0.003}] 

    if opt_method == 'Adam':
        #optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weightdecay)
        
    elif opt_method == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)


    for epoch in range(0, num_epoch):
        #print(epoch)
        since  = time.time()
        tr_loss, s0_arr, s1_arr, s2_arr, w1, w2, s0_2_arr, cnn_att, s_edge = train_model(model, train_loader, optimizer, scheduler, device, opt, num_epoch, writer)
        if epoch == num_epoch-1:
          #print(torch.Tensor.size(s0_arr))
          #print(np.shape(s0_arr))
          #np.save('s0_arr_' + name + '.npy', s0_arr)
          #np.save('s0_2_arr_' + name + '.npy', s0_2_arr)
          np.save('/data/hsm/FAGNN/quadrant_attention/results/s_edge_arr_' + name + '_' + str(current_fold) + '.npy', s_edge)
          #print(s0_arr)
        tr_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = evaluate_acc(model, train_loader, device, opt, writer, num_epoch) # train acc
        val_acc, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = evaluate_acc(model, val_loader, device, opt, writer, num_epoch) # val acc
        val_loss = test_loss(model, val_loader, device, opt, writer, num_epoch)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                           tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        history_train[0,epoch] = tr_loss
        history_train[1,epoch] = val_loss
        history_train[2,epoch] = tr_acc
        history_train[3,epoch] = val_acc


        #print(s1_arr)
        #print(s2_arr)
        #writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        #writer.add_histogram('Hist/hist_s2', s2_arr, epoch)



        if  epoch == num_epoch-1:
            print("saving model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))

    #######################################################################################
    ######################### Testing on testing set ######################################
    #######################################################################################

    if opt.load_model:
        model = FAGNN(opt.indim,opt.ratio,opt.nclass).to(device)
        model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
        model.eval()
        preds = []
        correct = 0
        for data in val_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos,data.pheno,data.behav)
            pred = outputs[0].max(1)[1]
            preds.append(pred.cpu().detach().numpy())
            correct += pred.eq(data.y).sum().item()
        preds = np.concatenate(preds,axis=0)
        trues = val_dataset.data.y.cpu().detach().numpy()
        cm = confusion_matrix(trues,preds)
        print("Confusion matrix")
        print(classification_report(trues, preds))

    else:
       model.load_state_dict(best_model_wts)
       model.eval()
       test_accuracy, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr = test_acc(val_loader)
       test_l= test_loss(val_loader,0)
       print("===========================")
       print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
       print(opt)

    
    #np.save('/data/hsm/FAGNN/quadrant_attention/results/train_history_'+ name + '_' + str(current_fold) + '.npy', history_train)
    
    #top_regions_E2, top_regions_E3, top_regions_E4 = node_finder(name, s0_arr)


    return len(val_index), tr_acc, test_accuracy, rmse, mad_mean, mad_median, r_squared, gamma, y_pred, y_arr, name#, top_regions_E2, top_regions_E3, top_regions_E4 






if __name__ == '__main__':
    num_sub = 170
    num_folds = 5
    total_runs = 10

    base_name = "FAGNN_quadrant_days_comp5_4"  

    grand_train_accuracies = []  # Store all train accuracies for all 10 cycles
    grand_test_accuracies = []   # Store all test accuracies for all 10 cycles

    for run in range(total_runs):
        all_train_accuracies = []
        all_test_accuracies = []

        for i in range(num_folds):

            name = base_name + f"_run_{run+1}_fold_{i+1}"  # Adding run and fold string to name
            results = np.zeros([8, num_folds])
            train_ind_2, test_ind_2 = train_val_test_split(num_sub, num_folds, i)

            results[0,1], results[1,1], results[2,1], results[3,1], results[4,1], results[5,1], results[6,1], results[7,1], y_pred, y_arr, _ = run_fagnn(num_sub, num_folds, i, train_ind_2, test_ind_2, name)

            train_accuracy = results[1, 1]
            test_accuracy = results[2, 1]

            all_train_accuracies.append(train_accuracy)
            all_test_accuracies.append(test_accuracy)

            print(f'Run {run + 1} Fold {i + 1}:')
            print('Train accuracy (MAE):', train_accuracy)
            print('Test accuracy (MAE):', test_accuracy)

            # Saving results
            np.save(name + '_history.npy', results)
            np.save(name + '_y_pred.npy', y_pred)
            np.save(name + '_y_arr.npy', y_arr)

            print('----------')

        # Append current run accuracies to grand list
        grand_train_accuracies.extend(all_train_accuracies)
        grand_test_accuracies.extend(all_test_accuracies)

        # Print all accuracies after each 10-run cycle
        print(f"Run {run + 1} Train Accuracies (MAE):", all_train_accuracies)
        print(f"Run {run + 1} Test Accuracies (MAE):", all_test_accuracies)
































