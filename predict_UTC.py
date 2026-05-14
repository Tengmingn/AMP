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
    name = 'UTC'
    mode = 'budget_2000'

    parser = argparse.ArgumentParser(description='SASS Framework')
    parser.add_argument('--resume_model', type=str,
                        default='/media/llog/AGMM-SASSdata/model_files/UTCsa/scribble0718_originalloss/DINOV2b_78.89.pth')
    parser.add_argument('--config', type=str, default='/home/lab532/llog-AMP-TGRS/AMP-TGRS/configs/Lima.yaml')
    parser.add_argument('--save-mask-path', type=str, default='/media/llog/testcode') 
    args = parser.parse_args()
    return args


def get_dataset(cfg):
    if cfg['dataset'] == 'treecanopy':
        valset = TreeDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'saopaulo':
        valset = Sao_PauloDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'santiago':
        valset = SantiagoDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    
    elif cfg['dataset'] == 'manaus':
        valset = ManausDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    
    elif cfg['dataset'] == 'lima':
        valset = LimaDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'caracas':
        valset = CaracasDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'buenos_aires':
        valset = Buenos_AiresDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'brasilia':
        valset = BrasiliaDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'bogota':
        valset = BogotaDataset(cfg['dataset'], cfg['data_root'], 'val', None)    

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
        for img, id in tbar:
            img = img.cuda()
            if cfg['dataset'] == 'treecanopy':
                pred = pre_slide(model, img, num_classes=cfg['nclass'],
                                tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)
            else:
                pred = ms_test(model, img)
               
            pred = torch.argmax(pred, dim=1)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred = Image.fromarray(pred, mode='P')
            pred.putpalette(cmap)
            filename_with_ext = os.path.basename(id[0])  
            filename_without_ext, ext = os.path.splitext(filename_with_ext)  
            new_filename = filename_without_ext + '.png'
            pred.save('%s/%s' % (args.save_mask_path, new_filename))

    # mIOU *= 100.0


if __name__ == '__main__':
    args = parse_args()

    print()
    print(args)
    main(args)
