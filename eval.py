from dataset.sass import *
from dataset.transform import normalize_back
from model.semseg.AMP import AMP
from util.utils import *
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

MODE = None


def parse_args():
    name = 'treetest'
    mode = 'scribble'

    parser = argparse.ArgumentParser(description='SASS Framework')
    parser.add_argument('--resume_model', type=str,
                        default='/home/lab532/llog/AGMM-SASS-difresnet/exp/treecanopy/scribble/reins_dinov2b_76.19.pth')
    parser.add_argument('--config', type=str, default='/media/llog/llog-AMP-TGRS/AMP-TGRS/configs/TreeTest.yaml')
    parser.add_argument('--save-mask-path', type=str, default="/media/llog/AGMM-SASSdata/testdata/Dinov2_predict/reins_dinov2b_scribble_miou76.19")
    args = parser.parse_args()
    return args


def get_dataset(cfg):
    if cfg['dataset'] == 'pascal':
        valset = VocDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'cityscapes':
        valset = CityDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'treecanopy':
        valset = TreeDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    
    elif cfg['dataset'] == 'treetest':
        valset = TreeTestDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'treetest2':
        valset = TreeTest2Dataset(cfg['dataset'], cfg['data_root'], 'val', None)
    else:
        valset = None

    return valset


def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = AMP(cfg, aux=False)
    checkpoint = torch.load(args.resume_model)
    model.load_state_dict(checkpoint, strict=False)
    print('\nParams: %.1fM' % count_params(model))
    model = model.cuda()
    model.eval()

    if not os.path.exists(args.save_mask_path):
        os.makedirs(args.save_mask_path)

    dataset = get_dataset(cfg)
    valloader = DataLoader(dataset, batch_size=1,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    tbar = tqdm(valloader)
    metric = meanIOU(num_classes=cfg['nclass'])
    cmap = color_map(cfg['dataset'])

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            if cfg['dataset'] == 'treetest' or cfg['dataset'] == 'treetest2':
                pred = pre_slide(model, img, num_classes=cfg['nclass'],
                                 tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)
            else:
                pred = ms_test(model, img)
               
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            mIOU = metric.mIoUevaluate()[-1]

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred = Image.fromarray(pred, mode='P')
            palette_data = [0, 0, 0, 255, 255, 255]
            pred.putpalette(palette_data)
            pred.save('%s/%s' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1])))

    mIOU *= 100.0


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
