import sys
import os
import numpy as np
import cv2
import time
from datetime import timedelta
import torch
from model import TASED_v2
from loss import KLDLoss
from dataset import DHF1KDataset, InfiniteDataLoader
from itertools import islice
import matplotlib.pyplot as plt

def main():
    ''' concise script for training '''
    # optional two command-line arguments
    path_indata = 'Atari_dataset'
    path_output = 'output'
    if len(sys.argv) > 1:
        path_indata = sys.argv[1]
        if len(sys.argv) > 2:
            path_output = sys.argv[2]

    # we checked that using only 2 gpus is enough to produce similar results
    num_gpu = 1
    pile = 5
    batch_size = 1
    num_iters = 1000
    len_temporal = 32
    file_weight = 'TASED_updated.pt'
    path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = TASED_v2()

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if 'base.' in name:
                bn = int(name.split('.')[1])
                sn_list = [0, 5, 8, 14]
                sn = sn_list[0]
                if bn >= sn_list[1] and bn < sn_list[2]:
                    sn = sn_list[1]
                elif bn >= sn_list[2] and bn < sn_list[3]:
                    sn = sn_list[2]
                elif bn >= sn_list[3]:
                    sn = sn_list[3]
                name = '.'.join(name.split('.')[2:])
                name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    # parameter setting for fine-tuning
    params = []
    for key, value in dict(model.named_parameters()).items():
        if 'convtsp' in key:
            params += [{'params':[value], 'key':key+'(new)'}]
        else:
            params += [{'params':[value], 'lr':0.01, 'key':key}] #0.001

    optimizer = torch.optim.SGD(params, lr=0.3, momentum=0.9, weight_decay=2e-7) #lr = 0.1
    criterion = KLDLoss()

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    torch.backends.cudnn.benchmark = False
    model.train()

    train_loader = InfiniteDataLoader(DHF1KDataset(path_indata, len_temporal), batch_size=batch_size, shuffle=True, num_workers=8)

    loss_statistic = []
    averaged_loss_statistic = []
    index_statistic = []

    i, step = 0, 0
    loss_sum = 0
    start_time = time.time()
    for clip, annt in islice(train_loader, num_iters*pile):
        with torch.set_grad_enabled(True):
            output = model(clip.cuda())
            loss = criterion(output, annt.cuda())

        loss_sum += loss.item()
        loss.backward()
        if (i+1) % pile == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # whole process takes less than 3 hours
            print ('iteration: [%4d/%4d], loss: %.4f, %s' % (step, num_iters, loss_sum/pile, timedelta(seconds=int(time.time()-start_time))), flush=True)

            loss_statistic.append(loss_sum/pile)
            if step%10==0:
                averaged_loss_statistic.append(sum(loss_statistic[step-10:step])/10)
                index_statistic.append(step)

            loss_sum = 0
            # adjust learning rate
            if step in [750, 950]:
                for opt in optimizer.param_groups:
                    if 'new' in opt['key']:
                        opt['lr'] *= 0.1   #0.1

            if step % 25 == 0:
                torch.save(model.state_dict(), os.path.join(path_output, 'iter_%04d.pt' % step))

        i += 1
    torch.save(model.state_dict(), os.path.join(path_indata, 'Atari_weight_file'))

    print('plotten')
    plt.plot(loss_statistic)
    plt.ylabel('loss')
    plt.savefig(os.path.join(path_indata, "loss.png"))

    plt.plot(index_statistic, averaged_loss_statistic)
    plt.ylabel('averaged loss')
    plt.savefig(os.path.join(path_indata, "averaged_loss.png"))


if __name__ == '__main__':
    main()