# coding: utf-8

from itertools import cycle

import fire
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch_train_test_loop import LoopComponent, TrainTestLoop

from data import Modality, get_dataset_splits
from models import MultiModalClassifier



### Globals

DEVICE = 'cuda:0'
DEFAULT_N_WORKERS = 8
LOG = print  # logging func


### Training code.


class CycleSampler(torch.utils.data.Sampler):
    """
    Samples indexes without replacement, cycling over all indexes, with
    optional shuffling indexes prior to each cycle.
    """
    def __init__(self, dataset, n, shuffle=False):
        def get_fresh_idxs(dataset):
            return torch.randperm(len(dataset)).tolist() if shuffle else range(len(dataset))
        self.cycled_idxs = cycle(idx for _ in cycle('forever') for idx in get_fresh_idxs(dataset))
        self.n = n

    def __iter__(self):
        return iter([next(self.cycled_idxs) for _ in range(self.n)])

    def __len__(self):
        return self.n


class LoopMain(LoopComponent):

    def __init__(self, modality, n_classes, parent_model, filename_prefix):
        self.modality = modality if modality in list(Modality) else getattr(Modality, modality)
        self.n_classes, self.parent_model, self.filename_prefix = (n_classes, parent_model, filename_prefix)
        self.mixup_dist = torch.distributions.beta.Beta(*([0.65 if modality == Modality.TEXT else 0.2] * 2))
        self.max_lr = 1e-3

    def on_train_begin(self, loop):
        loop.optimizer = torch.optim.RAdam(loop.model.parameters(), lr=self.max_lr)
        loop.scheduler = torch.optim.lr_scheduler.OneCycleLR(loop.optimizer, total_steps=loop.n_epochs * len(loop.train_data), max_lr=self.max_lr, pct_start=0.3)

    def on_epoch_begin(self, loop):
        if loop.is_training:
            self.saved_epoch_data = []

    def on_grads_reset(self, loop):
        loop.optimizer.zero_grad(set_to_none=True)

    def on_forward_pass(self, loop):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                if self.modality == Modality.TEXT:
                    texts, labels = (loop.batch['text'], loop.batch['label'].to(DEVICE))
                    tokenized = self.parent_model.tokenizer(texts, padding=True, truncation=True, return_overflowing_tokens=True, return_tensors="pt")
                    samples = { kwd: tokenized[kwd].to(DEVICE) for kwd in ['input_ids', 'attention_mask', 'overflow_to_sample_mapping'] }
                else:  # Modality.IMGS
                    samples, labels = (loop.batch[0].to(DEVICE), loop.batch[1].to(DEVICE))

                samples = self.parent_model.body(samples)
                targets = F.one_hot(labels, num_classes=self.n_classes).to(samples.dtype)  # [batch sz, n_classes]

                if loop.is_training:
                    r = self.mixup_dist.sample((labels.shape[0],)).to(device=DEVICE, dtype=samples.dtype)
                    idx = [*range(1, targets.shape[0]), 0]
                    samples = samples.lerp(samples[idx], r[:, None, None, None])
                    targets = targets.lerp(targets[idx], r[:, None])

        scores = loop.model(samples)
        with torch.no_grad():
            accuracy = (scores.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean()

        loop.scores, loop.targets, loop.accuracy = (scores, targets, accuracy)

    def on_loss_compute(self, loop):
        loop.loss = F.cross_entropy(loop.scores, loop.targets)

    def on_backward_pass(self, loop):
        loop.loss.backward()

    def on_optim_step(self, loop):
        loop.optimizer.step()
        loop.scheduler.step()

    def on_batch_end(self, loop):
        n_samples = len(loop.targets)
        param_group = loop.optimizer.param_groups[0]
        self.saved_epoch_data.append({
            'n_samples': n_samples,
            'epoch_desc': loop.epoch_desc,
            'epoch_num': loop.epoch_num,
            'batch_num' : loop.batch_num,
            'accuracy': loop.accuracy.item(),
            'loss': loop.loss.item(),
            'lr': param_group['lr'],
            'momentum': param_group['betas'][0],
        })

    def on_epoch_end(self, loop):
        if loop.is_validating:
            checkpoint = { 
                'weights': loop.model.state_dict(),
                'optimizer': loop.optimizer.state_dict(),
                'scheduler': loop.scheduler.state_dict(),
            }
            torch.save(checkpoint, f'{self.filename_prefix}_epoch{loop.epoch_num}.checkpoint')
        _ext = 'testing_data' if loop.is_testing else 'training_data'
        torch.save(self.saved_epoch_data, f'{self.filename_prefix}_epoch{loop.epoch_num}.{_ext}')


class LoopEpochStats(LoopComponent):

    def __init__(self, item_names=['loss', 'accuracy']):
        self.item_names = item_names

    def on_epoch_begin(self, loop):
        self.total, self.count = ({ name: 0.0 for name in self.item_names }, 0)
        self.pbar = tqdm(total=loop.n_batches, desc=f"{loop.epoch_desc} {loop.epoch_num}")

    def on_batch_end(self, loop):
        n = len(loop.targets)
        for name in self.item_names:
            x = getattr(loop, name)
            if torch.isfinite(x):
                self.total[name] += x.item() * n
            elif self.count > 0:
                self.total[name] += (self.total[name] / self.count) * n  # add prev mean
        self.count += n
        self.pbar.update(1)
        self.pbar.set_postfix_str(' '.join(['{}={:.4f}'.format(name, self.total[name] / self.count) for name in self.item_names]))

    def on_epoch_end(self, loop):
        self.pbar.close()



def train_head(
    dataset_name,
    n_epochs,
    batch_sz,
    n_iters_per_train_epoch=None,
    run_test=True,
    rng_seed=None,
    n_workers=DEFAULT_N_WORKERS,
    pretrained_path=None):

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    LOG('Getting dataset splits.')
    splits = get_dataset_splits(dataset_name)

    LOG('Creating dataloaders.')
    if n_iters_per_train_epoch is None:
        train_dl = torch.utils.data.DataLoader(splits['train'], batch_size=batch_sz, shuffle=True, num_workers=n_workers, pin_memory=True)
        n_iters_per_train_epoch = len(splits['train']) // batch_sz
    else:
        trn_sampler = CycleSampler(splits['train'], n=n_iters_per_train_epoch * batch_sz, shuffle=True)
        train_dl = torch.utils.data.DataLoader(splits['train'], batch_size=batch_sz, sampler=trn_sampler, num_workers=n_workers, pin_memory=True)

    valid_dl = torch.utils.data.DataLoader(splits['valid'], batch_size=batch_sz, shuffle=False, num_workers=n_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(splits['test'], batch_size=batch_sz, shuffle=False, num_workers=n_workers, pin_memory=True)

    LOG('Getting pretrained transformer and creating multimodal classifier.')
    parent_model = MultiModalClassifier(modality=splits['modality'], n_classes=splits['n_classes']).to(DEVICE)
    parent_model.transformer.eval()

    head = parent_model.head  # we will train only the head
    head_name = '{}_head_for_{}'.format(dataset_name, parent_model.transformer_config['name'])
    filename_prefix = 'checkpoints/{}'.format(head_name.replace('/', '-'))

    if pretrained_path is not None:
        LOG('---')
        LOG(f'Loading pretrained weights from checkpoint saved at: {pretrained_path}.')
        state_dict = torch.load(pretrained_path, map_location='cpu')['weights']
        state_dict = { k: v for k, v in state_dict.items() if not k.startswith('route.{}'.format(len(head.route) - 1)) }  # skip last routing
        head.load_state_dict(state_dict, strict=False)
        LOG('Freezing all layers except last routing.')
        for p in head.parameters():
            p.requires_grad = False
        for p in head.route[-1].parameters():
            p.requires_grad = True

    LOG('\nTraining "{}" for {} epochs, cycling over training data {:.1f} times.'.format(head_name, n_epochs, n_epochs * n_iters_per_train_epoch * batch_sz / len(splits['train'])))
    loop = TrainTestLoop(model=head, components=[
        LoopMain(modality=splits['modality'], parent_model=parent_model, n_classes=splits['n_classes'], filename_prefix=filename_prefix),
        LoopEpochStats()], train_data=train_dl, valid_data=valid_dl)
    loop.train(n_epochs=n_epochs)

    if run_test:
        LOG('\nTesting.')
        loop.test(test_dl)


if __name__ == '__main__':
    fire.Fire(train_head)
