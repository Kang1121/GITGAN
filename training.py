from utils import dataloader
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from models.Models import Models


def train(args):
    subs = args.sub_order
    kf = KFold(n_splits=args.num_cv)
    cv_set = np.arange(args.num_trials)
    for idx, test_subj in enumerate(subs):
        train_subjs = np.array(subs[idx+1:] + subs[:idx])
        for cv_index, (valid_index, test_index) in enumerate(kf.split(cv_set)):
            train_loader, train_sampler, _ = dataloader(args, train_subjs, valid_index, test_index, cv_set)
            model = Models(args)
            model.setup(args, test_subj, cv_index)
            loss_avg = 1
            for epoch in tqdm(range(args.epochs), desc="Epochs", unit="epoch"):
                train_sampler.set_epoch(epoch) if args.use_ddp else None
                batch_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} Batches", unit="batch", leave=False) if args.local_rank == 0 else train_loader
                loss_batch = 0
                for batch, data in enumerate(batch_loop):
                    model.set_input(data)
                    model.optimize_parameters(batch, loss_avg)
                    metrics = model.logging_info_train()
                    if args.local_rank == 0:
                        batch_loop.set_postfix(**{'Loss_' + k: "{:.3f}".format(v)[:5] for k, v in metrics['Loss'].items()},
                                               **{'Accuracy_' + k: "{:.3f}".format(v)[:5] for k, v in metrics['Accuracy'].items()})
                    loss_batch += metrics['Loss']['D']
                loss_avg = loss_batch / len(train_loader)
                model.update_learning_rate()
            model.save_networks(test_subj, cv_index, epoch)


def evaluate(args):
    subs = args.sub_order
    kf = KFold(n_splits=args.num_cv)
    cv_set = np.arange(args.num_trials)
    for idx, test_subj in enumerate(subs):
        train_subjs = np.array(subs[idx+1:] + subs[:idx])
        for cv_index, (valid_index, test_index) in enumerate(kf.split(cv_set)):
            _, _, test_loader = dataloader(args, train_subjs, valid_index, test_index, cv_set)
            model = Models(args)
            model.setup(args, test_subj, cv_index)
            batch_loop = tqdm(test_loader, desc=f"Batches", unit="batch", leave=False)
            for batch, data in enumerate(batch_loop):
                model.evaluate(data)
                metrics = model.logging_info_eval()
                batch_loop.set_postfix(**{'Accuracy_' + k: "{:.3f}".format(v)[:5] for k, v in metrics['Accuracy'].items()})


