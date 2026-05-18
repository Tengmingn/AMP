import argparse
from itertools import cycle
import logging
import os
import pprint
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
import wandb
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
# for backbone forzen use
from dataset.sass import *
from model.semseg.AMP import AMP
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import *
from util.PAR import PAR
from util.dist_helper import setup_distributed
from util.visualization_tools import get_pca_map, get_robust_pca
from util.mytool import *
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '28890'
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

# sh tools/train_voc.sh 3 28890

parser = argparse.ArgumentParser(description='Sparsely-annotated Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def choose_vis_channels(raw_features):
    ################## PCA part #################
    raw_features = raw_features[0].unsqueeze(0)
    hw = raw_features.shape[-2:]
    raw_features = raw_features.permute(0, 2, 3, 1)
    pca_feat = get_pca_map(raw_features, hw)
    return pca_feat


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)      
    logger = init_log('global', logging.INFO)
    logger.propagate = 0   
    rank, word_size = setup_distributed(port=args.port)
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    cudnn.enabled = True
    cudnn.benchmark = True
    model = AMP(cfg, aux=cfg['aux'])
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)      
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    ohem = False if cfg['criterion']['name'] == 'CELoss' else True 
    use_weight = True if cfg['dataset'] == 'cityscapes' else False

    kd_loss = KDLoss(T=10)
    par = PAR(dilations=[1,4,8,12,24],num_iter=10).cuda()

    trainset = TreeDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                              cfg['crop_size'], cfg['aug'])
    valset = TreeDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], None)

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size= 1 , pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    global_step = 0

    # hyperparameters
    beta1 = 0.5
    beta2 = 1/3

    # (Initialize logging)
    experiment = wandb.init(project='ourmethod', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=cfg['epochs'], batch_size=cfg['batch_size'], learning_rate=cfg['lr'],
             backbone=cfg['backbone'], img_scale=cfg['crop_size'], aug=cfg['aug'], mode = cfg['mode'])
    )

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        loss_m = AverageMeter()
        seg_m = AverageMeter()
        gmm_m = AverageMeter()
        k1_m = AverageMeter()
        pl_m = AverageMeter()
        trainsampler.set_epoch(epoch)

        for i, (img, mask, cls_label, id) in enumerate(trainloader):
            img, mask, cls_label = img.cuda(), mask.cuda(), cls_label.cuda()
            ori_feat, after_adapter, feat, outputs1, outputs2, pred = model(img)
            cls_loss = get_cls_loss(pred, cls_label, mask)
            # Gaussian
            cur_cls_label = build_cur_cls_label(mask, cfg['nclass'])
            pred_cl = clean_mask(pred, cls_label, True)
            vecs, proto_loss = cal_protypes(feat, mask, cfg['nclass'])
            res = GMM(feat, vecs, pred_cl, mask, cur_cls_label)
            gmm_loss = cal_gmm_loss(pred.softmax(1), res, cur_cls_label, mask) + proto_loss
            # Pseudo label part loss 
            if (epoch + 1)<=10:
                loss_ce1 = loss_calc(outputs1,mask,ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)
                loss_ce2 = loss_calc(outputs2,mask,ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)
                loss_ce3 = loss_calc(pred, mask,
                                 ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)                
                
                ave_output = (outputs1+outputs2)/2
                loss_kl1 = kd_loss(outputs1,ave_output.detach())
                loss_kl2 = kd_loss(outputs2,ave_output.detach())
                
                loss_pl = (loss_ce1 + loss_ce2) * beta2
                loss_kl = (loss_kl1 + loss_kl2)/2
                loss_seg = loss_ce3 * beta1 + cls_loss

            else:                
                outputs1_par = par(img,outputs1)
                outputs2_par = par(img,outputs2)
                res_par = par(img,res)
                merged_labels = select_3_confident_region(outputs1_par.detach(),outputs2_par.detach(),res_par.detach(),mask,thed=0.4)        

                loss_kl1 = joint_optimization(outputs1, outputs2.detach(), pred.detach(),10)
                loss_kl2 = joint_optimization(outputs2, outputs1.detach(), pred.detach(),10)
                                
                loss_ce1 = loss_calc(outputs1,mask,ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)
                loss_ce2 = loss_calc(outputs2,mask,ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)
                loss_ce3 = structure_loss(pred,merged_labels,cfg['nclass'])#

                loss_pl = (loss_ce1 + loss_ce2) * beta2
                loss_kl = (loss_kl1 + loss_kl2)/2
                loss_seg = loss_ce3 * beta1 + cls_loss
                
            # total loss
            loss = loss_seg + gmm_loss + loss_kl + loss_pl
            # for ablation
            # loss = loss_seg
            # loss = loss_seg + loss_pl
            # loss = loss_seg + loss_pl+ gmm_loss
            # loss = loss_seg + loss_pl+ loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), img.size()[0])
            seg_m.update(loss_seg.item(), img.size()[0])
            gmm_m.update(gmm_loss.item(), img.size()[0])
            k1_m.update(loss_kl.item(), img.size()[0])
            pl_m.update(loss_pl.item(), img.size()[0])

            iters += 1
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters:{:}, loss:{:.3f}, seg_loss:{:.3f}, '
                            'gmm_loss:{:.3f}, ''k1_loss:{:.3f}, ''pl_loss:{:.3f}'.format
                            (i, loss_m.avg, seg_m.avg, gmm_m.avg, k1_m.avg, pl_m.avg))
        
        global_step += 1
        ######### vis initialize #########
        vis_out_feat = choose_vis_channels(feat)
        vis_feat1 = choose_vis_channels(ori_feat[0])
        vis_feat2 = choose_vis_channels(ori_feat[1])
        vis_feat3 = choose_vis_channels(ori_feat[2])
        vis_feat4 = choose_vis_channels(ori_feat[3])

        after_adapter1 = choose_vis_channels(after_adapter[0])
        after_adapter2 = choose_vis_channels(after_adapter[1])
        after_adapter3 = choose_vis_channels(after_adapter[2])
        after_adapter4 = choose_vis_channels(after_adapter[3])

        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'images': wandb.Image(img[0].cpu()),
            'out_feats':wandb.Image(vis_out_feat),
            'feat1':wandb.Image(vis_feat1),
            'feat2':wandb.Image(vis_feat2),
            'feat3':wandb.Image(vis_feat3),
            'feat4':wandb.Image(vis_feat4),
            'after_adapter1':wandb.Image(after_adapter1),
            'after_adapter2':wandb.Image(after_adapter2),
            'after_adapter3':wandb.Image(after_adapter3),
            'after_adapter4':wandb.Image(after_adapter4),
            'masks': {
            'true': wandb.Image(mask[0].float().cpu()),
            'pred': wandb.Image(pred.argmax(dim=1)[0].float().cpu()),
            'pseudo label1':wandb.Image(outputs1.argmax(dim=1)[0].float().cpu()),
            'pseudo label2':wandb.Image(outputs2.argmax(dim=1)[0].float().cpu()),
            'GMM ouput':wandb.Image(res.argmax(dim=1)[0].float().cpu()),
            },
            'train loss': loss.item(),
            'step': global_step,
            'epoch': epoch
        })
        if cfg['dataset'] == 'treecanopy':
            eval_mode = 'center_crop'
        elif cfg['dataset'] == 'vaihingen':
            eval_mode = 'sliding_window'
        else:
            eval_mode = 'original'
        # print(f'eval mode is {eval_mode}')
        mIOU, iou_class, OA, PA, MPA, UA, MUA, Kappa = evaluate(model, valloader, eval_mode, cfg)
        F1_score = [round(100*2*PA*UA/(PA+UA), 2) for PA, UA in zip(PA, UA)] 
        mf1 = np.mean(F1_score)

        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}'.format(eval_mode, mIOU))
            logger.info('************************* ***** >>>> OA: {:.4f}'.format(OA))
            logger.info('************************* ***** >>>> PA: {}'.format(PA))
            logger.info('************************* ***** >>>> MPA: {:.4f}'.format(MPA))
            logger.info('************************* ***** >>>> UA: {}'.format(UA))
            logger.info('************************* ***** >>>> MUA: {:.4f}'.format(MUA))
            logger.info('************************* ***** >>>> mf1: {:.4f}'.format(mf1))
            logger.info('************************* ***** >>>> Kappa: {:.3f}\n'.format(Kappa))
            try:
                experiment.log({
                    'overall accuracy(OA)': OA,
                    'mean prodecer accuracy(MPA)': MPA,
                    'mean user accuracy(MUA)': MUA,
                    'Kappa consistency': Kappa,
                    'mF1 score': mf1,
                    'mean IoU': mIOU,
                    'step': global_step,
                    'epoch': epoch,
                })
            except:
                pass

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, now +'%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, now +'%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()

