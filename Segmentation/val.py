import torch
import argparse
import torch
from models.model1 import my_model1
from models.model2 import my_model2
from models.model3 import my_model3
import pickle
import os
import coco
from huffmancodec import HuffmanCodec
from torch.utils.data import DataLoader
from omegaconf import OmegaConf as omega
from tqdm import tqdm
import numpy as np

def get_activation(args, name):
    def hook(model, input, output):
        args.activation[name] = output.detach()
    return hook

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
def validate(args):
    count = 0
    count1 = 0 
    args.model.eval()
    args.evaluator.reset()

    with open(args.freq_file, 'rb') as outputfile:
        freq = pickle.load(outputfile)
    codec = HuffmanCodec.from_frequencies(freq)
    tbar = tqdm(args.val_loader)

    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.to(args.device), target.to(args.device)
            
            if (args.split_point == 'S1') or (args.split_point == 'US1'):
                output1 = args.model(image)
                output2 = args.activation['out1']
            else:    
                output1, output2 = args.model(image)
            
            codec_input = output2.cpu().detach().numpy()
                
            for j in range(codec_input.shape[0]):
                codec_input_j = codec_input[j,:].reshape(-1)
                encoded_bit = codec.encode(codec_input_j)
                count = count + len(encoded_bit)
                count1 = count1 + 1
            # measure miou
            pred = np.argmax(output1['out'].data.cpu().numpy(), axis=1)
            args.evaluator.add_batch(target.cpu().numpy(), pred)
            perf = args.evaluator.Mean_Intersection_over_Union()

        bpp = (count*8)/(count1*513*513)
        print("MIoU", perf)
        print("BPP", bpp)

    return perf

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--valdir', type=str, default='/data2/coco',
                        help='Val_Dir')
    parser.add_argument('--batch_size', type=int, default=40,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--split_point', type=str, default='S1',
                        metavar='N', help='split_point(default: auto)')
    parser.add_argument('--compress_factor', type=str, default='0',
                        metavar='N', help='compression_factor(default: auto)')
                                        
    args = parser.parse_args()
    args.device = "cuda:0" if args.gpu == True else "cpu"

    args.val_set = coco.COCOSegmentation(args, split='val', base_dir=args.valdir)

    args.val_loader = DataLoader(args.val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    args.evaluator = Evaluator(21)

    print('Split Point: ', args.split_point, ', Compression Factor: ', args.compress_factor)

    comp_dict = omega.load("dict.yaml")
    args.file_name = comp_dict[args.split_point][args.compress_factor]['file_name'] 
    args.freq_file = comp_dict[args.split_point][args.compress_factor]['freq_file']    
    checkpoint = torch.load(args.file_name)
     
    if (args.split_point == 'S1') or (args.split_point == 'US1'):
        args.model = my_model1(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])
        args.model.classifier[0].convs[1][3].load_state_dict(checkpoint['state_dict'], strict=False)
        args.activation = {}
        args.model.classifier[0].convs[1][3].quantizelayer.register_forward_hook(get_activation(args, 'out1'))
        
    elif (args.split_point == 'S2') or (args.split_point == 'US2'):
        args.model = my_model2(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])
        args.model.backbone.layer4[1].Bottleneck2.load_state_dict(checkpoint['state_dict'], strict=False)
    
    elif args.split_point == 'S3':
        args.model = my_model3(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])    
        args.model.backbone.layer4[0].Bottleneck2.load_state_dict(checkpoint['state_dict'], strict=False)
    
    args.model = args.model.to(args.device) 
    # evaluate on validation set
    validate(args)

if __name__ == "__main__":
   main()
