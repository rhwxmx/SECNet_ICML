import os
import sys
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
import provider_data
import time
import torch.nn.functional as F
from pathlib import Path
import sklearn.metrics as metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--train_method', type=str, default='others', help='DDP DP')
    parser.add_argument('--rank_dp', type=list, default=[0], help='rank')
    parser.add_argument('--batch_size', type=int, default = 80, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument("--data_path", type=str, default='./data/DVSGesture/', help="path to eyetracking_log")
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="path to tesnorboard_log")
    parser.add_argument("--log_name", type=str, default='/DVSGesture_32', help="path to tesnorboard_log")
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW,SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, testloader, criterion, device,mark_test,args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        label_seq = [[],[]]
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            # print(data.shape)
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()

            ######calculate the accuracy of all sequence######
            label_seq[0] = list([j for i in test_pred for j in i])
            label_seq[1] = list([j for i in test_true for j in i])
            from collections import Counter
            count = 0
            correct_seq= 0
            index = 0
            mark = mark_test if mark_test is not None else label_seq[1]
            for i in range(len(label_seq[1])-2):
                #### if mark is different, we run this code###
                if (mark[i] != mark[i+1]) or (i == len(label_seq[1])-2):
                    ####statistic the most common label in the sequence####
                    tar = Counter(label_seq[0][index:i+1])
                    tar = tar.most_common(1)[0][0]
                    ####if the most common label is equal to the label of the sequence, we count it####
                    correct_seq += 1 if tar==label_seq[1][i] else 0
                    index = i+1
                    count +=1
            count = -1 if count == 0 else count
            if args.train_method =="DDP":
                if dist.get_rank() == 0:
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_seq: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,correct_seq/count*100,correct_seq,count))
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_seq: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,correct_seq/count*100,correct_seq,count))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "time": time_cost,
        "test_acc_seq":correct_seq/count
    }

def train(net, trainloader, optimizer, criterion, device,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.reshape(args.batch_size,args.num_point,4)
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1) 
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()
        if args.train_method =="DDP":
            if dist.get_rank() == 0:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "time": time_cost
    }
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_model(net, epoch, path, acc, is_best, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    for key, value in kwargs.items():
        state[key] = value
    filepath = os.path.join(path, "last_checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best_checkpoint.pth'))

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(args.log_path + args.log_name)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/pointmlp.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    TRAIN_FILES = [args.data_path+"train.h5"]
    TEST_FILES = [args.data_path+"test.h5"]
    current_data_test, current_label_test, current_mark_test = provider_data.load_h5_mark(TEST_FILES[0])
    print("test",len(current_data_test),current_data_test.shape)
    current_label_test = np.squeeze(current_label_test)
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test.astype('int64'))
    current_data_test = current_data_test.reshape(-1,args.num_point,4)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    current_data_sum, current_label_sum,current_mark_sum = provider_data.load_h5_mark(TRAIN_FILES[0])
    current_label_sum = np.squeeze(current_label_sum) 
    print("train",current_data_sum.shape,current_label_sum.shape)
    current_data_sum = torch.from_numpy(current_data_sum)
    current_label_sum = torch.from_numpy(current_label_sum.astype('int64'))
    current_data_sum = current_data_sum.reshape(-1,args.num_point,4)
    dataset = torch.utils.data.TensorDataset(current_data_sum, current_label_sum)

    if args.train_method =="DDP":
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])  
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True,sampler = train_sampler)
    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    best_test_acc = 0.  # best test accuracy
    best_test_seq = 0.
    best_train_acc = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")


    from models.FECNet import FECNet
    classifier = FECNet(num_classes=args.num_category)
    criterion = cal_loss
    classifier.apply(inplace_relu)

    if args.train_method =="DDP":
        device = torch.device('cuda', local_rank) 
        classifier = classifier.to(local_rank) 
        classifier = DDP(classifier,device_ids=[local_rank],find_unused_parameters=True)
    elif args.train_method =="DP":
        device = 'cuda'
        classifier = classifier.to('cuda:0')
        classifier = torch.nn.DataParallel(classifier, device_ids=args.rank_dp)
    else:
        device = 'cuda'
        classifier = classifier.cuda()
        

    try:
        checkpoint = torch.load('last_checkpoint.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['net'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer =="SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer =="AdamW":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    global_epoch = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_out = train(classifier, trainDataLoader, optimizer, criterion, device,args)
        scheduler.step()
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        log_string('Train Accuracy: %f' % (train_out["acc"]))
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s")
        test_out = validate(classifier, testDataLoader, criterion, device,current_mark_test,args)
        if test_out["test_acc_seq"] > best_test_seq:
            best_test_seq = test_out["test_acc_seq"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc                    
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss            
        writer.add_scalar('Accuracy/test', test_out["acc"], epoch)
        writer.add_scalar('Accuracy/test_seq', test_out["test_acc_seq"], epoch)
        writer.add_scalar('Loss/test', test_out["loss"], epoch)
        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  
            best_train_acc=best_train_acc,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict())            
        print(f"Testing loss:{test_out['loss']}  "f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq*100.}%]\n\n")
        log_string('Test Accuracy: %f ' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))
        log_string('loss : %f' % train_out["loss"])
    if args.train_method =="DDP":
        dist.destroy_process_group()
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
