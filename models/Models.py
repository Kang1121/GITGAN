import models.networks as networks
import torch
import itertools


class Models(networks.BaseModel):

    # def __init__(self, args):
    #     super().__init__(args)
    #
    #     self.args = args
    #
    #     self.netE1 = networks.Encoder(args)
    #     self.netE2 = networks.Encoder(args)
    #     self.netG = networks.Generator(args)
    #     self.netD = networks.Discriminator(args)
    #
    #     self.model_names = ['E1', 'E2', 'G', 'D']
    #     self.loss_names = ['Cls', 'G', 'D']
    #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr_G)
    #     self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(),
    #                                                         self.netE1.parameters(),
    #                                                         self.netE2.parameters()),
    #                                         lr=self.args.lr_D)
    #
    #     self.optimizers = [self.optimizer_G, self.optimizer_D]
    #     self.criterionGAN = torch.nn.BCEWithLogitsLoss().to(self.device)
    #     self.criterionCls = torch.nn.NLLLoss().to(self.device)
    #
    #     self.update_counter = 0
    #     self.update_g_more = None
    #
    # def set_input(self, input):
    #     # (x1, y1, x2, y2)
    #     self.xs, self.ys = input[0].to(self.device), input[1].to(self.device)
    #     self.mixed_xs, self.mixed_ys1, self.mixed_ys2, self.lam \
    #         = self.mixup_data(self.xs, self.ys, device=self.device)
    #     self.xt, self.yt = input[2].to(self.device), input[3].to(self.device)
    #     self.outputs = {'zs': None, 'zt': None, 'zs_hat': None}
    #     self.hook_handle_zs = self.netE1.layers.classifier.register_forward_hook(lambda module, inp, out: self.generic_hook(inp, 'zs'))
    #     self.hook_handle_zt = self.netE1.layers.classifier.register_forward_hook(lambda module, inp, out: self.generic_hook(inp, 'zt'))
    #     self.hook_handle_zs_hat = self.netE2.layers.classifier.register_forward_hook(lambda module, inp, out: self.generic_hook(inp, 'zs_hat'))
    #
    #     self.ones = torch.ones(self.xt.shape[0]).to(self.device)
    #     self.zeros = torch.zeros(self.xt.shape[0]).to(self.device)

    # def forward(self):
    #     """
    #     Perform a forward pass on the networks.
    #     """
    #     # Forward pass on encoder and generator networks
    #     self.pred_zs, _ = self.netE1(self.mixed_xs), self.netE1(self.xt)
    #     self.xs_hat = self.netG(self.outputs['zs'].clone().detach())
    #     self.pred_zs_hat = self.netE2(self.xs_hat)
    #
    # def backward_G(self):
    #     self.lossCls = (self.mixup_criterion(self.criterionCls, self.pred_zs, self.mixed_ys1, self.mixed_ys2, self.lam)
    #                     + self.criterionCls(self.pred_zs_hat, self.ys))
    #
    #     self.lossG = self.criterionGAN(self.netD(self.outputs['zs_hat']), self.ones)
    #     loss = self.lossCls * 5 + self.lossG
    #     loss.backward()
    #
    # def backward_D(self):
    #     self.lossCls = (self.mixup_criterion(self.criterionCls, self.pred_zs, self.mixed_ys1, self.mixed_ys2, self.lam)
    #                     + self.criterionCls(self.pred_zs_hat, self.ys))
    #     self.lossD = self.criterionGAN(self.netD(self.outputs['zs']), self.zeros) + self.criterionGAN(self.netD(self.outputs['zs_hat']), self.zeros) + \
    #                 self.criterionGAN(torch.mean(self.netD(self.outputs['zt'])).expand_as(self.ones), self.ones)
    #     loss = self.lossCls * 5 + self.lossD
    #     loss.backward()
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._initialize_networks()
        self._initialize_optimizers()
        self._initialize_criteria()
        self.update_counter = 0
        self.update_g_more = None

    def _initialize_networks(self):
        """Initialize the various networks used in the model."""
        self.netE1 = networks.Encoder(self.args)
        self.netE2 = networks.Encoder(self.args)
        self.netG = networks.Generator(self.args)
        self.netD = networks.Discriminator(self.args)
        self.model_names = ['E1', 'E2', 'G', 'D']

    def _initialize_optimizers(self):
        """Initialize the optimizers for training."""
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr_G)
        params_to_optimize = itertools.chain(self.netD.parameters(), self.netE1.parameters(), self.netE2.parameters())
        self.optimizer_D = torch.optim.Adam(params_to_optimize, lr=self.args.lr_D)
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def _initialize_criteria(self):
        """Initialize the loss functions."""
        self.criterionGAN = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterionCls = torch.nn.NLLLoss().to(self.device)
        self.loss_names = ['Cls', 'G', 'D']

    def set_input(self, input):
        """Prepare the input data and set up necessary variables."""
        self._unpack_and_prepare_input(input)
        self._initialize_output_hooks()
        self.ones = torch.ones(self.xt.shape[0]).to(self.device)
        self.zeros = torch.zeros(self.xt.shape[0]).to(self.device)

    def _unpack_and_prepare_input(self, input):
        """Unpack and move the input data to the desired device."""
        self.xs, self.ys = input[0].to(self.device), input[1].to(self.device)
        self.mixed_xs, self.mixed_ys1, self.mixed_ys2, self.lam = self.mixup_data(self.xs, self.ys, device=self.device)
        self.xt, self.yt = input[2].to(self.device), input[3].to(self.device)
        self.outputs = {'zs': None, 'zt': None, 'zs_hat': None}

    def _initialize_output_hooks(self):
        """Register hooks to capture outputs from certain layers."""
        register_hook = lambda net, name: net.layers.classifier.register_forward_hook(lambda module, inp, out: self.generic_hook(inp, name))
        self.hook_handle_zs = register_hook(self.netE1, 'zs')
        self.hook_handle_zt = register_hook(self.netE1, 'zt')
        self.hook_handle_zs_hat = register_hook(self.netE2, 'zs_hat')

    def forward(self):
        """
        Perform a forward pass on the networks.
        """
        # Forward pass on encoder and generator networks
        self.pred_zs, _ = self.netE1(self.mixed_xs), self.netE1(self.xt)
        self.xs_hat = self.netG(self.outputs['zs'].clone().detach())
        self.pred_zs_hat = self.netE2(self.xs_hat)

    def backward_G(self):
        """
        Compute gradients for the generator.
        """
        # Compute classification loss for the generator
        self.lossCls = self._compute_classification_loss(self.pred_zs, self.pred_zs_hat)

        # Compute GAN loss for the generator
        self.lossG = self.criterionGAN(self.netD(self.outputs['zs_hat']), self.ones)

        # Combine the losses and compute gradients
        combined_loss = self.lossCls * 5 + self.lossG
        combined_loss.backward()

    def backward_D(self):
        """
        Compute gradients for the discriminator.
        """
        # Compute classification loss for the discriminator
        self.lossCls = self._compute_classification_loss(self.pred_zs, self.pred_zs_hat)

        # Compute GAN loss for the discriminator
        self.lossD = self._compute_discriminator_loss()

        # Combine the losses and compute gradients
        combined_loss = self.lossCls * 5 + self.lossD
        combined_loss.backward()

    def _compute_classification_loss(self, pred_zs, pred_zs_hat):
        """
        Helper method to compute the classification loss.
        """
        return (self.mixup_criterion(self.criterionCls, pred_zs, self.mixed_ys1, self.mixed_ys2, self.lam)
                + self.criterionCls(pred_zs_hat, self.ys))

    def _compute_discriminator_loss(self):
        """
        Helper method to compute the discriminator's GAN loss.
        """
        return (self.criterionGAN(self.netD(self.outputs['zs']), self.zeros)
                + self.criterionGAN(self.netD(self.outputs['zs_hat']), self.zeros)
                + self.criterionGAN(torch.mean(self.netD(self.outputs['zt'])).expand_as(self.ones), self.ones))

    def optimize_parameters(self, input, loss):
        """
        Optimize model parameters based on the provided input and loss.

        Args:
        - input: The model input.
        - loss: The current loss value.
        """
        self.train()

        # Initialize update_g_more if it's None based on the loss threshold.
        if self.update_g_more is None:
            self.update_g_more = loss < 5

        # Update the main network.
        self.update_network(self.update_g_more)
        self.update_counter += 1

        # If it's time to update the other network, reset the counter.
        if self.update_counter == self.args.update_freq_t:
            self.update_network(not self.update_g_more)
            self.update_counter = 0

        # Remove any previously set hooks.
        for handle in [self.hook_handle_zs, self.hook_handle_zt, self.hook_handle_zs_hat]:
            handle.remove()

    def update_network(self, update_g):
        """
        Update either generator or discriminator based on the flag update_g.

        Args:
        - update_g (bool): If True, updates the generator, otherwise the discriminator.
        """
        if update_g:
            self._update(self.netD, self.optimizer_G, self.backward_G, True)
            self._update(self.netG, self.optimizer_D, self.backward_D, False)
        else:
            self._update(self.netG, self.optimizer_D, self.backward_D, True)
            self._update(self.netD, self.optimizer_G, self.backward_G, False)

    def _update(self, net, optimizer, backward_method, flag=True):
        """
        Internal method to update a network's parameters.

        Args:
        - net: Network to be updated.
        - optimizer: Optimizer corresponding to the network.
        - backward_method: Backward method to compute gradients.
        - flag (bool, optional): If True, performs the optimizer step. Defaults to True.
        """
        # Set gradients of the discriminator based on the current network.
        self.set_requires_grad([self.netD], not net == self.netD)

        # Forward pass, compute gradients, and optionally update weights.
        self.forward()
        optimizer.zero_grad()
        backward_method()
        if flag:
            optimizer.step()

    def evaluate(self, input):
        """
        Evaluates the model on the given input.

        Args:
        - input (tuple): A tuple containing the input data (xt) and its corresponding labels (yt).

        Returns:
        - pred_zt (Tensor): Model predictions.
        """
        self.eval()  # Set the model to evaluation mode

        xt, yt = input
        pred_zt = self.netE2(xt)

        return pred_zt
