
import json
import optuna
import mlflow
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.model_utils import (RecDataset, LatentFactorModel, collate_fn)


def train_test_log_lfm(BATCH_SIZE, NUM_NEGATIVES, EDIM, 
                   EPOCH, OPTIMIZER_NAME, LR, counter=0, is_optuna=False, name=None):
    train_dataset = RecDataset(df_train['user_index'].values, df_train['node_index'], user2seen)


    dataloader = DataLoader(train_dataset, shuffle=True,num_workers=0, batch_size=BATCH_SIZE,
                            collate_fn=lambda x: collate_fn(x, NUM_NEGATIVES, max(df['node_index'].values)))
    
    
    model = LatentFactorModel(EDIM, user_indes, node_indes)
    
    optimizer = eval(f'torch.optim.{OPTIMIZER_NAME}')(model.parameters(), LR)
    
    bar = tqdm(total = EPOCH )
    
    for i in range(EPOCH):
        bar_loader = tqdm(total = len(dataloader) ,)
        losses = []
        for i in dataloader:
            users, items, labels = i
            optimizer.zero_grad()
            logits = model(users, items)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            loss.backward()
            optimizer.step()
            bar_loader.update(1)
            bar_loader.set_description(f'batch loss - {loss.item()}')
            losses.append(loss.item())
        
        bar.update(1)
        bar.set_description(f'epoch loss - {sum(losses)/len(losses)}')

    
    K = 30
    
    test_users = df_test['user_index'].unique()
    
    
    preds = model.pred_top_k(torch.tensor(test_users), K)[1].numpy()
    df_preds = pd.DataFrame({'node_index': list(preds), 'user_index': test_users, 'rank': [[j for j in range(0, K)]for i in range(len(preds))]})
    
    df_preds = df_preds.explode(['node_index', 'rank']).merge(
        df_test[['user_index', 'node_index']].assign(relevant=1).drop_duplicates(),
        on = ['user_index', 'node_index'],
        how='left' ,
    )
    df_preds['relevant'] = df_preds['relevant'].fillna(0)
    
    
    def calc_hitrate(df_preds, K):
        return  df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].max().mean()
    
    def calc_prec(df_preds, K):
        return  (df_preds[df_preds['rank']<K].groupby('user_index')['relevant'].mean()).mean()
        
    hitrate = calc_hitrate(df_preds, K)
    
    prec = calc_prec(df_preds, K)
    print(f'{hitrate=}, {prec=}, {K=}')

    if name is None:
        run_name = f'LFM_optuna_{counter}' if is_optuna else 'LFM custom'
    else:
        run_name = name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {
                'prec': prec, 
                'hitrate': hitrate,
            }
        )
        mlflow.log_params(
            {
                'model_name': model.__class__.__name__,
                'train_size': df_train.shape[0],
                'test_size':df_test.shape[0],
                'batch_size':BATCH_SIZE,
                'num_negatives':NUM_NEGATIVES,
                'edim':EDIM,
                'epoch':EPOCH,
                'optimizer_name': OPTIMIZER_NAME,
                'lr':LR
            }
        )
    print('ml flow ok')
    return prec, hitrate

def baseline_fit_optuna():
    counter = 0

    def objective(trial):
        nonlocal counter
        counter += 1
        BATCH_SIZE = trial.suggest_int("BATCH_SIZE", low=10_000, high=150_000, step=10_000)
        NUM_NEGATIVES = trial.suggest_int("NUM_NEGATIVES", low=3, high=10, step=2)
        EDIM = trial.suggest_int("EDIM", low=64, high=512, step=64)
        EPOCH = trial.suggest_int("EPOCH", low=8, high=15)
        OPTIMIZER_NAME = trial.suggest_categorical("OPTIMIZER_NAME", 
                                                   ['Adam', 'Adamax', 'RMSprop', 'SGD', 'Adadelta'])
        LR = trial.suggest_float("LR", 0.01, 1.5, step=0.1)
        prec, hitrate = train_test_log_lfm(BATCH_SIZE, NUM_NEGATIVES, EDIM, 
                                           EPOCH, OPTIMIZER_NAME, LR, is_optuna=True, counter=counter)
        return prec, hitrate

    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=5)

def baseline():
    bsize = 50_000
    nnegatives = 5
    edim = 128
    epoch = 10
    opt_name = 'Adam'
    lr = 1
    train_test_log_lfm(bsize, nnegatives, edim, 
                    epoch, opt_name, lr, is_optuna=False, name='baseline')

def custom_run(run_name='custom run', bsize=60000, nnegatives=7, edim=512, epoch=9, opt_name='RMSprop', lr=0.31):
    train_test_log_lfm(bsize, nnegatives, edim, 
                        epoch, opt_name, lr, is_optuna=False, name=run_name)

def best_model():
    bsize = 60000
    nnegatives = 7
    edim = 512
    epoch = 9
    opt_name = 'RMSprop'
    lr = 0.31000000000000005
    train_test_log_lfm(bsize, nnegatives, edim, 
                        epoch, opt_name, lr, is_optuna=False, name='best model LFM')

    
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Run recsys model with mlflow')
    parser.add_argument('mode', type=int, help='Operating modes:\n'
                        '1 - baseline,\n'
                        '2 - best run,\n'
                        '3 - fitting baseline using optuna,\n'
                        '4 - run with specific params (bsize, nnegatives, edim, epoch, opt_name, lr).'
                        )
    parser.add_argument('--run_name', type=str, 
                        default='custom run', help='Run name for mlflow (str, optional, default: from best model).\n'
                        'WARNING!!! This param uses only in mode 4.')
    parser.add_argument('--bsize', type=int, 
                        default=60000, help='Batch size (int, optional, default: from best model)')
    parser.add_argument('--nnegatives', type=int, 
                        default=7, help='Number of negative samples (int, optional, default: from best model)')
    parser.add_argument('--edim', type=int, 
                        default=512, help='Edim param (int, optional, default: from best model)')
    parser.add_argument('--epoch', type=int, 
                        default=9, help='Number of epochs (int, optional, default: from best model)')
    parser.add_argument('--opt_name', type=str, 
                        default='RMSprop', help='Optimizator name (str, optional, default: from best model).\n'
                        'MUST BE IN torch.optim. E.g.: Adadelta or ASGD.')
    parser.add_argument('--lr', type=float, 
                        default=0.31, help='Learning rate (float, optional, default: from best model)')
    args = parser.parse_args()

    with open('data/node2name.json', 'r') as f:
        node2name = json.load(f)

    node2name = {int(k):v for k,v in node2name.items()}

    df = pd.read_parquet('data/clickstream.parque')
    df = df.head(100_000)
    df['is_train'] = df['event_date'] < df['event_date'].max() - pd.Timedelta('2 day')
    df['names'] = df['node_id'].map(node2name)
    train_cooks = df[df['is_train']]['cookie_id'].unique()
    train_items = df[df['is_train']]['node_id'].unique()


    df = df[(df['cookie_id'].isin(train_cooks)) & (df['node_id'].isin(train_items))]
    user_indes, index2user_id = pd.factorize(df['cookie_id'])
    df['user_index'] = user_indes

    node_indes, index2node = pd.factorize(df['node_id'])
    df['node_index'] = node_indes
    df_train, df_test = df[df['is_train']], df[~df['is_train']]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)


    user2seen = df_train.groupby('user_index')['node_index'].agg(lambda x: list(set(x)))
    url = os.environ["MLFLOW_UTL"]
    mlflow.set_tracking_uri(f'http://{url}/')

    mlflow.set_experiment('homework-pipeline-ydnikolaev')
    
    if args.mode == 1:
        baseline()

    elif args.mode == 2:
        best_model()

    elif args.mode == 3:
        baseline_fit_optuna()

    elif args.mode == 4:
        optimizers = ['Adadelta', 'Adagrad', 'Adam', 
                    'AdamW', 'SparseAdam', 'Adamax', 
                    'ASGD', 'LBFGS', 'NAdam', 'RAdam', 
                    'RMSprop', 'Rprop', 'SGD']
        if args.opt_name not in optimizers:
            raise ValueError(f'Optimizer name must be in torch.optim ({optimizers}).')
        
        custom_run(run_name=args.run_name, bsize=args.bsize, nnegatives=args.nnegatives, edim=args.edim,
                   epoch=args.epoch, opt_name=args.opt_name, lr=args.lr)
    else:
        raise ValueError('Mode is not specified!!! Must be in [1, 2, 3, 4]. See --help for information.')
