import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from data import SampleGenerator
from neumf import NeuMF
from engine import Engine
import os
import warnings
import time
import random
import wandb
warnings.filterwarnings("ignore")

def main(): 
    wandb.init(project="recommendation")
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                type=float,
                default=0.001,
                help='learning rate')
    parser.add_argument('--epochs',
                type=int,
                default=100,
                help='learning rate')
    parser.add_argument('--batch_size',
                type=int,
                default=1024,
                help='train batch size')
    parser.add_argument('--latent_dim_mf',
                type=int,
                default=8,
                help='latent_dim_mf')
    parser.add_argument('--num_layers',
                type=int,
                default=3,
                help='num layers')
    parser.add_argument('--num_ng',
                type=int,
                default=4,
                help='negative sample')
    parser.add_argument('--ng_sample',
                type=int,
                default=99,
                help='test negative sample')
    parser.add_argument('--l2',
                type=float,
                default=0.0,
                help='l2_regularization')
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')
    args = parser.parse_args()
    wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    

    
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
   
    
    #NCF model
    model = NeuMF(num_users = len(user_id), num_items = len(item_id),
                  latent_dim_mf = args.latent_dim_mf, num_layers = args.num_layers)
    model.cuda()
    model = nn.DataParallel(model)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    wandb.watch(model)
    
    N = []
    patience = 0
    t1 = time.time()
    train_loader = sample_generator.instance_a_train_loader(args.num_ng, args.batch_size)
    t2 = time.time()
    print("Train_Loader Time = {:.4f}".format(t2-t1))
    for epoch in range(args.epochs):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        random.shuffle(train_loader.sampler.seq)
        model.train()
        total_loss = 0
        t3 = time.time()
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, ratings = batch[0], batch[1], batch[2]
            ratings = ratings.float()
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            optimizer.zero_grad()
            output = model(users, items)
            loss = criterion(output, ratings)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            wandb.log({'Batch Loss': loss})
            total_loss += loss
            
        print("Train 끝")
        engine = Engine()    
        hit_ratio, ndcg = engine.evaluate(model = model, evaluate_data= evaluate_data, epoch_id=epoch)
        wandb.log({"epoch" : epoch,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        N.append(ndcg)
        if patience > 10:
            print("Patience = 10 초과")
            print("ndcg = {:.4f}".format(max(N)))
            break
        elif N[-1] < max(N):
            patience += 1
            print("Patience = {} ndcg = {:.4f}".format(patience, max(N)))
        else:
            patience = 0
            print("Patience = {}".format(patience))
        t4 = time.time()
        print("Epoch Time = {:.4f}".format(t4-t3))
        #import ipdb; ipdb.set_trace()
        
if __name__ == '__main__':
    main()          