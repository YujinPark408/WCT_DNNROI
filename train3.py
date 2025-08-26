import sys
import os
import math
import itertools
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn    # For CUDA DNN 
import torch.nn as nn                   # Utils for building Networks
from torch import optim                 # For optimazation

# User-defined models 
from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

# User-defined util functions
from eval_util import eval_dice, eval_loss, eval_eff_pur
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, chw_to_hwc
from utils import h5_utils as h5u

# ==== Dummy functions ====
def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

def lr_exp_decay(optimizer, lr0, gamma, epoch):
    lr = lr0*math.exp(-gamma*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    
# Settings for random seed
#torch.manual_seed(42)
#np.random.seed(42)

# ==== Training function ====
def train_net(net,
              im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0'],
              ma_tags = ['frame_ductor0'],
              truth_th = 100,
			# file_img  = [f"data/g4-rec-r{i}.h5" for i in range(10)],
			  file_img  = [f"data/g4-rec-{i}_zero.h5" for i in [0,1,3]],
			# file_mask = [f"data/g4-tru-r{i}.h5" for i in range(10)],
			  file_mask = [f"data/g4-tru-{i}.h5" for i in [0,1,3]],
              sepoch=0,          # start epoch number
              nepoch=1,          # number of epochs
              strain=0,          # start sample for training
              ntrain=10,         # start sample for training
              sval=450,          # start sample for val
              nval=50,           # number of sample for val
              batch_size=10,     # batch size
              lr=0.1,            # learning-rate
              val_percent=0.10,  #
              save_cp=True,      # use Checkpoints saving 
              gpu=False,         # use cuda
              img_scale=0.5):    # 

    # Write your dir for saving checkpoints
    dir_checkpoint = 'checkpoints_unet_mini/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)


    # Dataset ID generation 
    iddataset_img = {}
    iddataset_mask = {}
    event_per_file = 10
    img_id_offset = 100 # event_zero_id_offset for img(reco data)
    mask_id_offset = 0 # event_zero_id_offset for mask(true data)

    def id_gen_img(index):
        return (index // event_per_file, index % event_per_file + img_id_offset)
        
    def id_gen_mask(index):
        return (index // event_per_file, index % event_per_file + mask_id_offset)
    
    iddataset_img['train'] = [id_gen_img(i) for i in list(strain+np.arange(ntrain))]
    iddataset_img['val'] = [id_gen_img(i) for i in list(sval+np.arange(nval))]

    iddataset_mask['train'] = [id_gen_mask(i) for i in list(strain+np.arange(ntrain))]
    iddataset_mask['val'] = [id_gen_mask(i) for i in list(sval+np.arange(nval))]

    # Open log file for training progress
    outfile_log = open(dir_checkpoint+'/log','a+')

    print(f"Training IDs (Img): {iddataset_img['train']}", file=outfile_log, flush=True)
    print(f"Training IDs (Mask): {iddataset_mask['train']}", file=outfile_log, flush=True)
    print(f"Validation IDs (Img): {iddataset_img['val']}", file=outfile_log, flush=True)
    print(f"Validation IDs (Mask): {iddataset_mask['val']}", file=outfile_log, flush=True)

    print(f'''
    Starting training:
        Epochs: {nepoch}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(iddataset_img['train'])}
        Validation size: {len(iddataset_img['val'])}
        Checkpoints: {str(save_cp)}
        CUDA: {str(gpu)}
    ''', file=outfile_log, flush=True)  
    
    N_train = len(iddataset_img['train'])

    # Optimizer setup
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Loss function (Binary Cross-Entropy Loss for segmentation)
    criterion = nn.BCELoss()

    print(f'''
    im_tags: {im_tags}
    ma_tags: {ma_tags}
    truth_th: {truth_th}
    ''', file=outfile_log, flush=True)
    
    # Open CSV files for logging metrics
    outfile_loss_batch = open(dir_checkpoint+'/loss-batch.csv','a+') # loss of each batch
    outfile_loss       = open(dir_checkpoint+'/loss.csv','a+')       # Mean loss of each  training epoch
    outfile_eval_dice  = open(dir_checkpoint+'/eval-dice.csv','a+')  # Dice coeff for validation(모델이 실제로 정확한 분할을 하는지 평가,0~1, 1에 가까울수록 좋음)
    outfile_eval_loss  = open(dir_checkpoint+'/eval-loss.csv','a+')  # Mean loos of valitadtion epoch
    
    # Evaluation image setup
    eval_labels = [
        '75-75',
        '87-85',
    ]
    eval_imgs = []
    eval_masks = []
    for label in eval_labels:
        eval_imgs.append('eval/eval-'+label+'/g4-rec-0.h5')
        eval_masks.append('eval/eval-'+label+'/g4-tru-0.h5')
    outfile_ep = []
    for label in eval_labels:
        outfile_ep.append(open(dir_checkpoint+'/ep-'+label+'.csv','a+'))
    
    # Load model checkpoint if sepoch > 0  
    if sepoch > 0 :
        net.load_state_dict(torch.load('{}/CP{}.pth'.format(dir_checkpoint, sepoch-1)))
    
    # Main training loop
    for epoch in range(sepoch,sepoch+nepoch):
        # scheduler = lr_exp_decay(optimizer, lr, 0.04, epoch)
        scheduler = optimizer
        
        print(f'epoch: {epoch} start')
        print(optimizer, file=outfile_log, flush=True)

        # Data loading parameters
        rebin = [1, 10] # Rebinning scales, rebin[0] : channel & rebin[1] : timeticks
        x_range = [800, 1600] # PDSP, V, left-closed right-open interval
        #x_range = [0, 1600] # PDSP, U & V, left-closed right-open interval
		# x_range = [476, 952] # PDVD, V
        y_range = [0, 600]
        z_scale = 4000

        print(f'''
        file_img: {file_img}
        file_mask: {file_mask}
        ''', file=outfile_log, flush=True)

        print(f'Starting epoch {epoch}/{nepoch}.')
        net.train()  # Set network to training mode

        # Prepare data iterators for training, validation, and evaluation
        train = zip(
          h5u.get_chw_imgs(file_img, iddataset_img['train'], im_tags, rebin, x_range, y_range, z_scale),
          h5u.get_masks(file_mask, iddataset_mask['train'], ma_tags, rebin, x_range, y_range, truth_th)
        )
        val = zip(
          h5u.get_chw_imgs(file_img, iddataset_img['val'], im_tags, rebin, x_range, y_range, z_scale),
          h5u.get_masks(file_mask, iddataset_mask['val'], ma_tags, rebin, x_range, y_range, truth_th)
        )
        eval_data = []
        for i in range(len(eval_imgs)):
            id_eval = [0]
            eval_data.append(
                zip(
                    h5u.get_chw_imgs(eval_imgs[i], id_eval,   im_tags, rebin, x_range, y_range, z_scale),
                    h5u.get_masks(eval_masks[i],   id_eval,   ma_tags, rebin, x_range, y_range, truth_th)
                )
            )

        epoch_loss = 0

        # Iterate over batches for training
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            # Optional: plot images/masks for debugging
            # if False:
            #     h5u.plot_mask(b[0][1])
            #     h5u.plot_img(chw_to_hwc(b[0][0]))

            # Convert numpy arrays to PyTorch tensors
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            # Move tensors to GPU if enabled
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Forward pass
            masks_pred = net(imgs)

            # Flatten predictions and true masks for BCELoss
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            # Calculate loss
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            # Print batch loss and log to file
            print(f'{epoch} : {i * batch_size / N_train:.4f} --- loss: {loss.item():.6f}')
            print(f'{i * batch_size / N_train:.4f}, {loss.item():.6f}', file=outfile_loss_batch, flush=True)
           
            # Backward pass and optimizer step
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            scheduler.step()  # Update model parameters
            # optimizer.step()

        # Calculate and print average epoch loss
        epoch_loss = epoch_loss / (i + 1)
        print('Epoch finished ! Loss: {:.6f}'.format(epoch_loss))
        print('{:.4f}, {:.6f}'.format(epoch, epoch_loss), file=outfile_loss, flush=True)

        # Save model checkpoint
        if save_cp:
            torch.save(net.state_dict(),
                      dir_checkpoint + 'CP{}.pth'.format(epoch))
            print('Checkpoint e{} saved !'.format(epoch))

        # Perform validation and evaluation (Always perform validation)
        if True:
            val1, val2 = itertools.tee(val, 2)
            
            # val_dice = eval_dice(net, val1, gpu)
            # print('Validation Dice Coeff: {:.4f}, {:.6f}'.format(epoch, val_dice))
            # print('{:.4f}, {:.6f}'.format(epoch, val_dice), file=outfile_eval_dice, flush=True)

            val_loss = eval_loss(net, criterion, val2, gpu)
            print('Validation Loss: {:.4f}, {:.6f}'.format(epoch, val_loss))
            print('{:.4f}, {:.6f}'.format(epoch, val_loss), file=outfile_eval_loss, flush=True)
            
            # # Efficiency and Purity evaluation (commented out in original)
            # for data, out in zip(eval_data,outfile_ep):
            #     ep = eval_eff_pur(net, data, 0.5, gpu)
            #     print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch, ep[0], ep[1], ep[2], ep[3]), file=out, flush=True)
            



def get_args():
    parser = OptionParser()
    parser.add_option('--start-epoch', dest='sepoch', default=0, type='int',
                      help='start epoch number')
    parser.add_option('-e', '--nepoch', dest='nepoch', default=1, type='int',
                      help='number of epochs')

    parser.add_option('--start-train', dest='strain', default=0, type='int',
                      help='start sample for training')
    parser.add_option('--ntrain', dest='ntrain', default=10, type='int',
                      help='start sample for training')
    parser.add_option('--start-val', dest='sval', default=450, type='int',
                      help='start sample for val')
    parser.add_option('--nval', dest='nval', default=50, type='int',
                      help='number of sample for val')

    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options



# ============================== Main ==============================
if __name__ == '__main__':
    args = get_args()

    # Set number of threads for Torch (important for CPU performance)
    torch.set_num_threads(17) 

    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    # im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23
#	ma_tags = ['frame_ductor0']
    ma_tags = ['frame_deposplat0']
    truth_th = 10

    # Initialize the U-Net model.
    net = UNet(len(im_tags), len(ma_tags))
    # net = UResNet(len(im_tags), len(ma_tags))
    # net = NestedUNet(len(im_tags),len(ma_tags))

    # Load pre-trained model if specified
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # Move model to GPU if enabled
    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    # Start training
    try:
        train_net(net=net,
                  im_tags=im_tags,
                  ma_tags=ma_tags,
                  truth_th=truth_th,
                  sepoch=args.sepoch,
                  nepoch=args.nepoch,
                  strain=args.strain,
                  ntrain=args.ntrain,
				  sval=args.sval,
                  nval=args.nval,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=False,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
