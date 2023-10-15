from abc import ABC, abstractmethod
import logging
import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import os
from torch.nn.functional import elu
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import kaiming_normal_init


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    elif opt.lr_policy == 'warmup':
        scheduler_steplr = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=scheduler_steplr)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.

    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape)) or
            (self._pool_weights.is_cuda != x.is_cuda) or
            (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(
                np.ones(weight_shape, dtype=np.float32) / float(n_pool)
            )
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled


class Generator(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Define norm_layer and other constants here
        self.norm_layer = get_norm_layer(norm_type='instance')
        self.use_dropout = False
        self.use_bias = True  # False when use batchnorm

        if args.dataset == 'OpenBMI':
            self.layers = self.build_architecture_a()
        elif args.dataset in ['BCICIV2a', 'BCICIV2a+']:
            self.layers = self.build_architecture_b()
        else:
            raise ValueError(f"Unsupported dataset name: {args.dataset}")

    def block(self, in_channels, out_channels, kernel_size, stride, padding, transposed=True):
        layers = []
        if transposed:
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.use_bias),
                self.norm_layer(out_channels),
                nn.ReLU(True)
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.use_bias),
                self.norm_layer(out_channels),
                nn.ReLU(True)
            ]

        if out_channels != 1:  # Exclude the last Conv layer from normalization and activation
            layers.extend([self.norm_layer(out_channels), nn.ReLU(True)])
        return layers

    def build_architecture_a(self):
        layers = nn.Sequential(
            # Temporal
            *self.block(200, 512, (6, 1), 1, 0),
            *self.block(512, 512, (6, 1), (2, 1), 0),
            *self.block(512, 512, (6, 1), (2, 1), 0),
            *self.block(512, 512, (6, 1), (2, 1), 0),
            *self.block(512, 300, (3, 1), (2, 1), 0),
            *self.block(300, 300, (3, 1), (2, 1), 0),
            *self.block(300, 100, (4, 1), (2, 1), 0),
            # Spatial
            *self.block(100, 100, (1, 3), (1, 2), 0),
            *self.block(100, 50, (1, 4), (1, 2), 0),
            *self.block(50, 50, (1, 4), (1, 2), 0),
            *self.block(50, 30, (1, 5), (1, 2), 0),
            *self.block(30, 1, (1, 1), 1, 0, transposed=False)
        )

        return layers

    def build_architecture_b(self):
        layers = nn.Sequential(
            # Temporal
            *self.block(200, 300, (5, 1), (1, 1), 0),
            *self.block(300, 300, (5, 1), (2, 1), 0),
            *self.block(300, 300, (6, 1), (2, 1), 0),
            *self.block(300, 300, (6, 1), (2, 1), 0),
            *self.block(300, 200, (8, 1), (2, 1), 0),
            *self.block(200, 200, (8, 1), (2, 1), 0),
            *self.block(200, 200, (12, 1), (2, 1), 0),
            # Spatial
            *self.block(200, 100, (1, 4), (1, 2), 0),
            *self.block(100, 100, (1, 4), (1, 2), 0),
            *self.block(100, 50, (1, 4), (1, 2), 0),
            *self.block(50, 1, (1, 1), 1, 0, transposed=False)
        )

        return layers

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):

    def __init__(self, args):
        super().__init__()

        input_sizes = {
            'OpenBMI': 1400,
            'BCICIV2a': 800,
            'BCICIV2a+': 800
        }

        self.layers = self._build_architecture(input_sizes[args.dataset])

    def _linear_block(self, in_features, out_features, dropout=0.25):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def _build_architecture(self, input_size):
        return nn.Sequential(
            self._linear_block(input_size, 500),
            self._linear_block(500, 500),
            self._linear_block(500, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1)).squeeze()


class Encoder(nn.Sequential):
    """Deep ConvNet model from Schirrmeister et al 2017.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        args,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_conv_nonlin=elu,
        first_pool_mode="max",
        later_pool_mode="max",
        drop_prob=0.5,
        split_first_layer=True,
        instance_norm=True,
        stride_before_pool=False,
    ):
        super().__init__()
        self.in_chans = args.channels
        self.n_classes = args.classes
        self.input_window_samples = args.samples
        self.final_conv_length = 'auto'
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_conv_nonlin
        self.first_pool_mode = first_pool_mode
        self.later_pool_mode = later_pool_mode
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.instance_norm = instance_norm
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            self.conv_stride = self.pool_time_stride
            self.pool_stride = 1
        else:
            self.conv_stride = 1
            self.pool_stride = self.pool_time_stride
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        self.first_pool_class = pool_class_dict[self.first_pool_mode]
        self.later_pool_class = pool_class_dict[self.later_pool_mode]

        # First layer
        if self.split_first_layer:
            self.layers = nn.Sequential(
                nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1)),
                nn.Conv2d(self.n_filters_time, self.n_filters_spat, (1, self.in_chans), stride=(self.conv_stride, 1), bias=not self.instance_norm),
                nn.InstanceNorm2d(self.n_filters_spat, affine=True) if self.instance_norm else nn.Identity(),
                nn.ReLU(),
                self.first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(self.pool_stride, 1))
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_chans, self.n_filters_time, (self.filter_time_length, 1), stride=(self.conv_stride, 1), bias=not self.instance_norm),
                nn.InstanceNorm2d(self.n_filters_time, affine=True) if self.instance_norm else nn.Identity(),
                nn.ReLU(),
                self.first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(self.pool_stride, 1))
            )

        # Additional layers
        filter_configs = [
            (self.n_filters_time if self.split_first_layer else self.n_filters_spat, self.n_filters_2, self.filter_length_2),
            (self.n_filters_2, self.n_filters_3, self.filter_length_3),
            (self.n_filters_3, self.n_filters_4, self.filter_length_4)
        ]
        for idx, (n_filters_before, n_filters, filter_length) in enumerate(filter_configs, start=2):
            self.layers.add_module(f"layer{idx}", self._conv_pool_block(n_filters_before, n_filters, filter_length))

        # Create a dummy input to infer the shape of the layers
        self._dummy_input = torch.ones(1, 1, self.input_window_samples, self.in_chans)
        out = self.layers(self._dummy_input)
        self.final_conv_length = out.shape[-2]

        # Classifier
        self.layers.add_module("classifier", nn.Sequential(
            nn.Conv2d(self.n_filters_4, self.n_classes, (self.final_conv_length, 1)),
            nn.LogSoftmax(dim=1)
        ))

    def _conv_pool_block(self, n_filters_before, n_filters, filter_length):
        return nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(n_filters_before, n_filters, (filter_length, 1), stride=(self.conv_stride, 1), bias=not self.instance_norm),
            nn.InstanceNorm2d(n_filters, affine=True) if self.instance_norm else nn.Identity(),
            nn.ReLU(),
            self.later_pool_class(kernel_size=(self.pool_time_length, 1), stride=(self.pool_stride, 1))
        )

    def forward(self, x):
        return self.layers(x).squeeze()


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, args):
        """Initialize the BaseModel class.

        Parameters:
            args (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.args = args
        self.isTrain = args.train
        self.device = args.local_rank
        self.save_dir = 'ckpts/{}'.format(self.args.dataset) # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.outputs = {}
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self, input, loss):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def generic_hook(self, output, key):
        self.outputs[key] = output[0]

    def setup(self, opt, test_subj, cv_index):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    net.apply(kaiming_normal_init)
                    net.to(self.device)
                    DDP(net, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True) if opt.use_ddp else None
        else:
            self.load_networks(test_subj, cv_index, opt.epochs-1)

        if opt.local_rank == 0:
            self.print_networks(opt.verbose)

    def mixup_data(self, x, y, alpha=1.0, device=None):
        """Returns mixed inputs, pairs of targets, and lambda"""

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if device:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']

        return lr

    def get_current_losses(self):
        """Return training losses/errors."""
        return {name: float(getattr(self, 'loss' + name)) for name in self.loss_names}

    def get_current_accuracy_train(self):
        """Return training accuracy."""
        predictions = {
            'zs': self.pred_zs,
            'zs_hat': self.pred_zs_hat
        }
        return {key: float(torch.sum(torch.argmax(pred, dim=1) == self.ys).item() / self.ys.size(0)) for key, pred in predictions.items()}

    def get_current_accuracy_eval(self):
        predictions = {
            'zt': self.pred_zt
        }
        return {key: float(torch.sum(torch.argmax(pred) == self.yt).item() / self.yt.size(0)) for key, pred in predictions.items()}

    def logging_info_train(self):
        """Return a combined dictionary of current losses and accuracies."""
        metrics = {
            'Loss': self.get_current_losses(),
            'Accuracy': self.get_current_accuracy_train()
        }
        return metrics

    def logging_info_eval(self):
        """Return a combined dictionary of current losses and accuracies."""
        metrics = {
            'Accuracy': self.get_current_accuracy_eval()
        }
        return metrics

    def save_networks(self, sub_idx, fold, epoch):
        """Save all the networks to the disk.

        Parameters:
            sub_idx (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            save_filename = 'sub%s_net%s_cv%s_epo%s.pth' % (sub_idx, name, fold, epoch)
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                save_path = os.path.join(self.save_dir, save_filename)
                torch.save(net.cpu().state_dict(), save_path, _use_new_zipfile_serialization=False)

    def load_networks(self, sub_idx, fold, epoch):
        """Load all the networks from the disk.

        Parameters:
            sub_idx (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for model_name in self.model_names:
            if isinstance(model_name, str):
                load_filename = 'sub%s_net%s_cv%s_epo%s.pth' % (sub_idx, model_name, fold, epoch)
                load_path = 'ckpts/{}/{}'.format(self.args.dataset, load_filename)

                net = getattr(self, 'net' + model_name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                logging.info('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location='cpu')
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        logging.info('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    logging.info(net)
                logging.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        logging.info('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
