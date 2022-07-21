import json
import os
import random
import time
from datetime import timedelta
import torch
from config import get_config
from test import test
from train import train
from utils.init_util import init_dir, create_valid_data_loader
from valid import valid
from model import build_net


def main(args):
    print('----------------------------------------------------------------------')
    print(args.model_name)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid':
        log = open(os.path.join(args.log, 'log.txt'), 'w')
        # loader valid data
        valid_loader = create_valid_data_loader(args)

        net = build_net(args.model_name, args.image_channel)
        state_dict = torch.load(args.test_model)
        net.load_state_dict(state_dict)
        net = net.to(args.device)
        valid(args, net, log, valid_loader)
    elif args.mode == 'test':
        log = open(os.path.join(args.log, 'log.txt'), 'w')
        net = build_net(args.model_name, args.image_channel)

        if str(args.device) == 'cpu':
            print('hello')
            # resume = torch.load(args.test_model, map_location='cpu')
            # net.load_state_dict(resume['model'])
            state_dict = torch.load(args.test_model, map_location='cpu')
        else:
            state_dict = torch.load(args.test_model)
        net.load_state_dict(state_dict)
        net = net.to(args.device)
        test(args, net, log)


if __name__ == '__main__':
    args = get_config()
    # print(args)

    # Init dir
    init_dir(args)

    main(args)
