import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random, wandb

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA, MANIQA3Stages, MANIQA_new_vit, MANIQA_new_vit_or_ceo
from config import Config
from utils.process2 import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process2 import split_dataset_kadid10k, split_dataset_koniq10k
from utils.process2 import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from loss import ceo_loss

from sklearn.metrics import f1_score, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, or_criterion, optimizer, scheduler, train_loader):
    losses = []
    or_losses = []
    total_losses = []
    pred_label = []
    total_target = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    # for data in tqdm(train_loader):
    running_loss, or_running_loss, total_running_loss = .0, .0, .0
    with tqdm(desc='Epoch %d - training' % epoch, unit='it', total=len(train_loader)) as pbar:
        for it, data in enumerate(train_loader):
            x_d = data['d_img_org'].cuda()
            score = data['score']
            target = data['label'].long().cuda()

            score = torch.squeeze(score.type(torch.FloatTensor)).cuda()
            pred_d, label = net(x_d)

            optimizer.zero_grad()
            loss = criterion(torch.squeeze(pred_d), score)
            ordinal_loss = or_criterion(torch.squeeze(label), target)
            total_loss = loss + ordinal_loss

            losses.append(loss.item())
            or_losses.append(ordinal_loss.item())
            total_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred_d.data.cpu().numpy()
            labels_batch_numpy = score.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            running_loss += float(loss)
            or_running_loss += float(ordinal_loss)
            total_running_loss += float(total_loss)

            comparison_label = np.transpose(np.linspace(0.0, 20.0, num=21).reshape(-1,1).repeat(len(label), axis=1), (1, 0))
            pred_label.append(np.argmin(abs((label.cpu().detach() - comparison_label)), axis=1))
            total_target.append(target)

            pbar.set_postfix(running_loss=running_loss / (it + 1), or_running_loss=or_running_loss / (it + 1),
                             total_running_loss=total_running_loss / (it + 1))
            pbar.update()

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    or_ret_loss = np.mean(or_losses)
    total_ret_loss = np.mean(total_losses)

    acc = accuracy_score([element.cpu().detach() for sublist in total_target for element in sublist],
                         [element.cpu().detach() for sublist in pred_label for element in sublist])
    f1 = f1_score([element.cpu().detach() for sublist in total_target for element in sublist],
                  [element.cpu().detach() for sublist in pred_label for element in sublist], average='macro')

    logging.info('train epoch:{} / loss:{:.4} / or_ret_loss:{:.4} / total_ret_loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} ===== acc:{:.4} ===== f1:{:.4}'.
                 format(epoch + 1, ret_loss, or_ret_loss, total_ret_loss, rho_s, rho_p, acc, f1))

    return ret_loss, or_ret_loss, total_ret_loss, rho_s, rho_p, acc, f1


def eval_epoch(config, epoch, net, criterion, or_criterion, test_loader):
    with torch.no_grad():
        losses = []
        or_losses = []
        total_losses = []
        pred_label = []
        total_target = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        running_loss, or_running_loss, total_running_loss = 0., 0., 0.
        with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(train_loader)) as pbar:
            for it, data in enumerate(test_loader):
                pred = 0
                label = []
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    score = data['score']
                    target = data['label'].long().cuda()

                    score = torch.squeeze(score.type(torch.FloatTensor)).cuda()

                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    m_pred, label = net(x_d)
                    pred += m_pred

                pred /= config.num_avg_val

                # compute loss
                loss = criterion(torch.squeeze(m_pred), score)

                ordinal_loss = or_criterion(torch.squeeze(label), target)
                total_loss = loss + ordinal_loss

                losses.append(loss.item())
                or_losses.append(ordinal_loss.item())
                total_losses.append(total_loss.item())

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = score.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)

                running_loss += float(loss)
                or_running_loss += float(ordinal_loss)
                total_running_loss += float(total_loss)

                comparison_label = np.transpose(
                    np.linspace(0.0, 20.0, num=21).reshape(-1, 1).repeat(len(label), axis=1), (1, 0))
                pred_label.append(np.argmin(abs((label.cpu().detach() - comparison_label)), axis=1))
                total_target.append(target)

                pbar.set_postfix(running_loss=running_loss / (it + 1), or_running_loss=or_running_loss / (it + 1),
                                 total_running_loss=total_running_loss / (it + 1))
                pbar.update()


        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        acc = accuracy_score([element.cpu().detach() for sublist in total_target for element in sublist] , [element.cpu().detach() for sublist in pred_label for element in sublist])
        f1 = f1_score([element.cpu().detach() for sublist in total_target for element in sublist], [element.cpu().detach() for sublist in pred_label for element in sublist], average='macro')

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== or loss:{:.4} ===== total loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.
            format(epoch + 1, np.mean(losses), np.mean(or_losses), np.mean(total_losses), rho_s, rho_p))

        print('Epoch:{} ===== loss:{:.4} ===== or loss:{:.4} ===== total loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== acc:{:.4} ===== f1:{:.4}'.
            format(epoch + 1, np.mean(losses), np.mean(or_losses), np.mean(total_losses), rho_s, rho_p, acc, f1))

        return np.mean(losses), np.mean(or_losses), np.mean(total_losses), rho_s, rho_p, acc, f1


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(42)

    wandb.login(key='e476bb91d83495c4174473429b20f197416f08c9')
    wandb_runner = wandb.init(
        project="MANIQA_OR",
        entity="ljc",
        job_type="train",
        name='MANIQA_OR_Joint',
    )

    # config file
    config = Config({
        # dataset path
        "dataset_name": "samsung",

        # samsung
        "samsung_train_path": "/data/train_mos.csv",
        "samsung_valid_path": "/data/valid_mos.csv",
        
        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1, # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8,
        
        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 384,
        "prob_aug": 0.7,

        # model
        "patch_size": 16,
        "img_size": 384,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        
        # load & save checkpoint
        "model_name": "samsung",
        "type_name": "samsung",
        "ckpt_path": "/data2/samsung_IC/MANIQA_new_vit_or_joint2/",               # directory for saving checkpoint
        "log_path": "./data2/samsung_IC/MANIQA_new_vit_or_joint2/log/",
        "log_file": ".log",
        "tensorboard_path": "/data2/samsung_IC/MANIQA_new_vit_or_joint2/tensorboard/"
    })
    
    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)


    if  config.dataset_name == 'samsung':
        from data_name.SAMSUNG import samsung_ordinal
        dis_train_path = config.samsung_train_path
        dis_val_path = config.samsung_valid_path
        Dataset = samsung_ordinal
    else:
        pass

    # data load
    train_dataset = Dataset.samsung_ordinal(
        csv_file=dis_train_path,
        root='../samsung_LK99/dataset/train/',
        transform=transforms.Compose([
            Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config.prob_aug), ToTensor()]),
    )
    val_dataset = Dataset.samsung_ordinal(
        csv_file=dis_val_path,
        root='../samsung_LK99/dataset/train/',
        transform=transforms.Compose([
            Normalize(0.5, 0.5), ToTensor()]),
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=False)


    # model defination
    net = MANIQA_new_vit_or_ceo(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    # path_model = '/data2/samsung_IC/MANIQA/ckpt_koniq10k.pt'
    # net.load_state_dict(torch.load(path_model), strict=False)

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    criterion2 = ceo_loss.CEOLoss(num_classes=13)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, or_ret_loss, total_ret_loss, rho_s, rho_p, acc, f1 = train_epoch(epoch, net, criterion, criterion2,
                                                                          optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        wandb.log({"Train SRCC": rho_s, "Train PLCC": rho_p, "Epoch": epoch, "Train loss": loss_val,
                   "Train or_loss": or_ret_loss, "Train total_loss": total_ret_loss,
                   "Train reg acc": acc, "Train reg f1": f1})

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))

            loss, or_loss, total_loss, rho_s, rho_p, acc, f1 = eval_epoch(config, epoch, net, criterion, criterion2, val_loader)
            wandb.log({"validation SRCC": rho_s, "validation PLCC": rho_p, "validation loss": loss,
                       "validation or loss": or_loss, "validation total loss": total_loss,
                       "validation reg acc": acc, "validation reg f1": f1})

            logging.info('Eval done...')

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                best_srocc = rho_s
                best_plcc = rho_p

                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

    wandb.finish()