import os
import sys
import time
import math
import models
import tabulate
from utils import utils
from utils.utils import VisionTask, NLPTask
import logging

import torch
from tensorboardX import SummaryWriter

def main():
    args = utils.get_arguments()
    # set up the repository
    logging.info('Preparing directory %s' % args.dir)
    os.makedirs(args.dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    logging.info('Loading dataset %s from %s' % (args.dataset, args.data_path))
    dataloaders = utils.create_dataset(args)

    logging.info('Using model %s' % args.model)
    model = utils.create_model(args)

    logging.info('Creating criterion')
    criterion = utils.create_criterion(args)

    logging.info('Creating metric')
    metrics = utils.create_metric(args)

    logging.info('Creating optimizer %s' % args.optim)
    optimizer = utils.create_optimizer(args, model)

    if args.job_type == 'vision':
        task = VisionTask(args, dataloaders, model, criterion, optimizer, metrics)
    elif args.job_type == 'nlp':
        task = NLPTask(args, dataloaders, model, criterion, optimizer, metrics)

    if args.use_tensorboard:
        assert args.tensorboard_dir is not None
        tb_writer = SummaryWriter(args.tensorboard_dir)

    logging.info('Start the main training loop')

    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    utils.save_checkpoint(
        args.dir,
        args.start_epoch,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

    columns = ['ep'] + ['train/'+m for m in metrics['train'].keys()] + ['test/'+m for m in metrics['test'].keys()]

    for epoch in range(args.start_epoch, args.epochs):
        task.adjust_learning_rate(epoch)
        train_res = task.train_epoch()
        for metric in metrics['train']:
            metrics['train'][metric] = train_res[metric]
            tb_writer.add_scalar('train/'+metric, train_res[metric], epoch)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = task.eval()
            for metric in metrics['train']:
                metrics['test'][metric] = test_res[metric]
                tb_writer.add_scalar('test/'+metric, test_res[metric], epoch)
        else:
            test_res = {'loss': None, 'accuracy': None}

        if (epoch + 1) % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
        
        values = [epoch + 1] + [m for m in metrics['train'].values()] + [m for m in metrics['test'].values()]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
    tb_writer.close()
    logging.info('Finished training')

if __name__ == "__main__":
    main()