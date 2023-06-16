import torch
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from models.model1 import mymodel1
from models.model2 import mymodel2
from models.model3 import mymodel3
import pickle
import os
from huffmancodec import HuffmanCodec
from omegaconf import OmegaConf as omega
from tqdm import tqdm
    
def validate(args):
    count = 0
    count1 = 0 
    top1 = AverageMeter('Acc@1', ':6.2f')   
    args.model.eval()
    
    with open(args.freq_file, 'rb') as outputfile:
        freq = pickle.load(outputfile)

    codec = HuffmanCodec.from_frequencies(freq)
    tbar = tqdm(args.val_loader)

    with torch.no_grad():
        for i, (images, target) in enumerate(tbar):
            images = images.to(args.device)
            target = target.to(args.device)


            output1, output2 = args.model(images)
            codec_input = output2.cpu().detach().numpy()    
            for j in range(codec_input.shape[0]):
                codec_input_j = codec_input[j,:].reshape(-1)
                encoded_bit = codec.encode(codec_input_j)
                count = count + len(encoded_bit)
                count1 = count1 + 1

            # measure accuracy and record loss
            acc1, _ = accuracy(output1, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            
        bpp = (count*8)/(count1*224*224)
        print("Accuracy :", float(top1.avg))
        print("BPP :", bpp)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--valdir', type=str, default='/data1/imagenet/val',
                        help='Val_Dir')
    parser.add_argument('--batch_size', type=int, default=384,
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

    normalize1 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    args.val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize1,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)                                       
    
    comp_dict = omega.load("dict.yaml")
    args.file_name = comp_dict[args.split_point][args.compress_factor]['file_name'] 
    args.freq_file = comp_dict[args.split_point][args.compress_factor]['freq_file']    
    checkpoint = torch.load(args.file_name)

    print('Split Point: ', args.split_point, ', Compression Factor: ', args.compress_factor)     
    if (args.split_point == 'S1') or (args.split_point == 'US1'):
        args.model = mymodel1(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])
        args.model.Bottleneck2.load_state_dict(checkpoint['state_dict'], strict=False)
    
    elif (args.split_point == 'S2') or (args.split_point == 'US2'):
        args.model = mymodel2(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])
        args.model.layer4[0].Bottleneck2.load_state_dict(checkpoint['state_dict'], strict=False)
    
    elif args.split_point == 'S3':
        args.model = mymodel3(comp_dict[args.split_point][args.compress_factor]['qp'],comp_dict[args.split_point][args.compress_factor]['out_channel'],comp_dict[args.split_point][args.compress_factor]['stride_val'])    
        args.model.layer3[5].Bottleneck3.load_state_dict(checkpoint['state_dict'], strict=False)
    
    args.model = args.model.to(args.device) 
    # evaluate on validation set
    validate(args)

if __name__ == "__main__":
   main()
