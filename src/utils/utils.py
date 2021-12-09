import os
from io import open
import argparse
from models import vgg, rnn
import math

import torch.onnx
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .nlp_helper import *
#from vision_helper import *


def get_arguments():
    parser = argparse.ArgumentParser(description='hyperparameter deception experiment')
    
    parser.add_argument('--dir', type=str, default=None, required=True, 
                        help='training directory (default: None)')
    parser.add_argument('--job_type', type=str, default='vision',
                        help='job type (vision or nlp)')
    parser.add_argument('--job_name', type=str, default='test',
                        help='job name')
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--data_path', type=str, default="./data", required=True, metavar='PATH',
                        help='path to datasets location (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers (default: 4) used in the data loader')
    parser.add_argument('--model', type=str, default='vgg16_bn', required=True, metavar='MODEL',
                        help='model name (default: vgg16_bn), type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--optim', type=str, default='SGD', required=True,
                        help='optimizer name (default: SGD)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='the epoch to start training (default: 0)')
    parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                        help='save frequency (default: 25)')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                        help='evaluation frequency (default: 5)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='whether to use tensorboard in the visualization')
    parser.add_argument('--tensorboard_dir', type=str, default=None, 
                        help='tensorboard directory')
    
    # system related arguments
    parser.add_argument('--use_cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--use_data_parallel', action='store_true',
                        help='use data parallelism for faster training')
    
    # NLP related arguments
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    args = parser.parse_args()
    return args

def create_model(args):
    if 'vgg' in args.model:
        model = vgg.__dict__[args.model]()
    elif 'LSTM' in args.model or args.model == 'Transformer':
        ntokens = len(args.corpus.dictionary)
        if args.model == 'Transformer':
            model = rnn.TransformerModel(ntokens,
                                        args.emsize,
                                        args.nhead,
                                        args.nhid,
                                        args.nlayers,
                                        args.dropout)
        else:
            model = rnn.RNNModel(args.model,
                                ntokens,
                                args.emsize,
                                args.nhid,
                                args.nlayers,
                                args.dropout,
                                args.tied)
    elif 'logistic_regression' in args.model:
        model = torch.nn.Linear(784, 10)
        model.weight.data.fill_(0.)
        model.bias.data.fill_(0.)
    else:
        raise NotImplementedError
    if args.use_cuda:
        model = model.cuda()
    return model

def create_dataset(args):
    if args.job_type == 'vision':
        if 'CIFAR' in args.dataset:
            ds = getattr(torchvision.datasets, args.dataset)
            path = os.path.join(args.data_path, args.dataset.lower())
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            train_set = ds(path, train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # Note that in (Wilson et al., 2017), random crop is not used,
                # for consistency we disable this operation.
                # transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize
                ]))
            test_set = ds(path, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ]))
        elif 'MNIST' in args.dataset:
            path = os.path.join(args.data_path, args.dataset.lower())
            train_set = torchvision.datasets.MNIST(path,
                train=True, 
                transform=transforms.Compose([transforms.ToTensor(),]),
                download=True)
            test_set = torchvision.datasets.MNIST(path,
                train=False, 
                transform=transforms.Compose([transforms.ToTensor(),]))
        loaders = {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        }

    else:
        args.corpus = Corpus(os.path.join(args.data_path, args.dataset))
        args.ntokens = len(args.corpus.dictionary)
        loaders = {
            'train':args.corpus.train,
            'valid':args.corpus.valid,
            'test':args.corpus.test
        }

    return loaders

def create_optimizer(args, model):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    elif args.optim == 'HB':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd
        )
    elif args.optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    elif args.optim == 'RMSProp':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
    else:
        raise NotImplementedError
    return optimizer

def create_criterion(args):
    if args.job_type == 'vision':
        criterion = F.cross_entropy
    elif args.job_type == 'nlp':
        criterion = torch.nn.NLLLoss()
    else:
        raise NotImplementedError
    return criterion

def create_metric(args):
    metrics = dict()
    if args.job_type == 'vision':
        metrics = {
            'train':{
                'accuracy' : 0.,
                'loss'    : 0.
            },
            'test':{
                'accuracy' : 0.,
                'loss'    : 0.
            },
        }
    elif args.job_type == 'nlp':
        metrics = {
            'train':{
                'perplexity' : 0.,
                'loss'    : 0.
            },
            'test':{
                'perplexity' : 0.,
                'loss'    : 0.
            },
        }
    else:
        raise NotImplementedError
    return metrics

def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class Task(object):
    def adjust_learning_rate(self):
        raise NotImplementedError
    
    def train_epoch(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError


class VisionTask(Task):
    def __init__(self, args, dataloaders, model, criterion, optimizer, metrics):
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
    
    def adjust_learning_rate(self, epoch):
        factor = 0.5**(epoch // 25)
        lr = self.args.lr * factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_epoch(self):
        loss_sum = 0.0
        correct = 0.0

        self.model.train()

        for i, (input, target) in enumerate(self.dataloaders['train']):
            if self.args.dataset == 'MNIST':
                input = input.view(-1, 28*28)
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_var.data.view_as(pred)).sum().item()

        return {
            'loss': loss_sum / len(self.dataloaders['train'].dataset),
            'accuracy': correct / len(self.dataloaders['train'].dataset) * 100.0,
        }

    def eval(self):
        loss_sum = 0.0
        correct = 0.0

        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.dataloaders['test']):
                if self.args.dataset == 'MNIST':
                    input = input.view(-1, 28*28)
                input = input.cuda()
                target = target.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                output = self.model(input_var)
                loss = self.criterion(output, target_var)

                loss_sum += loss.item() * input.size(0)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target_var.data.view_as(pred)).sum().item()

        return {
            'loss': loss_sum / len(self.dataloaders['test'].dataset),
            'accuracy': correct / len(self.dataloaders['test'].dataset) * 100.0,
        }

class NLPTask(Task):
    def __init__(self, args, dataloaders, model, criterion, optimizer, metrics):
        self.args        = args
        self.dataloaders = dataloaders
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.metrics     = metrics

        self.train_data  = self.batchify(self.dataloaders['train'], self.args.batch_size)
        self.valid_data  = self.batchify(self.dataloaders['valid'], self.args.batch_size)
        self.test_data   = self.batchify(self.dataloaders['test'],  self.args.batch_size)

    def repackage_hidden(self, h):
        r"""Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def get_batch(self, source, i):
        r"""
        get_batch subdivides the source data into chunks of length args.bptt.
        If source is equal to the example output of the batchify function, with
        a bptt-limit of 2, we'd get the following two Variables for i = 0:
        ┌ a g m s ┐ ┌ b h n t ┐
        └ b h n t ┘ └ c i o u ┘
        Note that despite the name of the function, the subdivison of data is not
        done along the batch dimension (i.e. dimension 1), since that was handled
        by the batchify function. The chunks are along dimension 0, corresponding
        to the seq_len dimension in the LSTM.
        """
        seq_len = min(self.args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if self.args.use_cuda:
            data = data.cuda()
        return data
    
    def adjust_learning_rate(self, epoch):
        pass

    def train_epoch(self):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        ntokens = len(self.args.corpus.dictionary)
        if self.args.model != 'Transformer':
            hidden = self.model.init_hidden(self.args.batch_size)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.args.bptt)):
            data, targets = self.get_batch(self.train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()
            self.optimizer.zero_grad()
            if self.args.model == 'Transformer':
                output = self.model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = self.repackage_hidden(hidden)
                output, hidden = self.model(data, hidden)
            loss = self.criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            #for p in model.parameters():
            #    p.data.add_(p.grad, alpha=-lr)
            self.optimizer.step()
            total_loss += loss.item()

        return {
            'perplexity': math.exp(total_loss / (len(self.train_data) - 1)),
            'loss': total_loss / (len(self.train_data) - 1)
        }

    def eval(self):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        ntokens = len(self.args.corpus.dictionary)
        if self.args.model != 'Transformer':
            hidden = self.model.init_hidden(self.args.batch_size)
        with torch.no_grad():
            for i in range(0, self.valid_data.size(0) - 1, self.args.bptt):
                data, targets = self.get_batch(self.valid_data, i)
                if self.args.model == 'Transformer':
                    output = self.model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = self.model(data, hidden)
                    hidden = self.repackage_hidden(hidden)
                total_loss += len(data) * self.criterion(output, targets).item()
        return {
            'perplexity': math.exp(total_loss / (len(self.valid_data) - 1)),
            'loss': total_loss / (len(self.valid_data) - 1)
        }