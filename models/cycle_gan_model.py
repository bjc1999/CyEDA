import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'


    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionEdge = networks.EdgeLoss(use_cuda=True)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            self.netG_B.train()
        else:
            self.netG_A.eval()
            self.netG_B.eval()
        print('-----------------------------------------------')


    def train(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()


    def eval(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    
    def test(self):
        pass


    def predict(self, id):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)

            if self.opt.n_mask > 0:
                self.fake_B, self.latent_real_A, self.mask_A, self.conv9feat_A = self.netG_A.forward(self.real_A)
                self.fake_A, self.latent_real_B, self.mask_B, self.conv9feat_B = self.netG_B.forward(self.real_B)
            else:
                self.fake_B = self.netG_A.forward(self.real_A)
                self.fake_A = self.netG_B.forward(self.real_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)

        return [(f'real_A_{id}', real_A), (f'fake_B_{id}', fake_B), (f'real_B_{id}', real_B), (f'fake_A_{id}', fake_A)]

    
    def get_image_paths(self):
        return self.image_paths


    def backward_D_basic(self, netD, real, fake):
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A.backward()
    

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B.backward()
    

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        if self.opt.n_mask > 0:
            self.fake_B, self.latent_real_A, self.mask_A, self.conv9feat_A = self.netG_A.forward(self.real_A)
            self.fake_A, self.latent_real_B, self.mask_B, self.conv9feat_B = self.netG_B.forward(self.real_B)
            self.rec_A, _, _, _ = self.netG_B.forward(self.fake_B)
            self.rec_B, _, _, _ = self.netG_A.forward(self.fake_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)
            self.fake_A = self.netG_B.forward(self.real_B)
            self.rec_A = self.netG_B.forward(self.fake_B)
            self.rec_B = self.netG_A.forward(self.fake_A)
        

    def backward_G(self):
        pred_fake_B = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake_B, True)

        pred_fake_A = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake_A, True)

        if self.opt.cycle_loss == 'Dis':
            pred_rec_B = self.netD_A.forward(self.rec_B)
            self.loss_Cycle_A = self.criterionGAN(pred_rec_B, True) * self.opt.lambda_cycle
            pred_rec_A = self.netD_B.forward(self.rec_A)
            self.loss_Cycle_B = self.criterionGAN(pred_rec_A, True) * self.opt.lambda_cycle
        elif self.opt.cycle_loss == 'L1':
            self.loss_Cycle_A = self.criterionCycle(self.real_A, self.rec_A) * self.opt.lambda_cycle
            self.loss_Cycle_B = self.criterionCycle(self.real_B, self.rec_B) * self.opt.lambda_cycle
        elif self.opt.cycle_loss == 'Edge':
            self.real_A_edge, self.rec_A_edge, self.loss_Cycle_A = self.criterionEdge(self.real_A, self.fake_B)
            self.real_B_edge, self.rec_B_edge, self.loss_Cycle_B = self.criterionEdge(self.real_B, self.fake_A)
            self.loss_Cycle_A = self.loss_Cycle_A * self.opt.lambda_cycle 
            self.loss_Cycle_B = self.loss_Cycle_B * self.opt.lambda_cycle
        elif self.opt.cycle_loss == 'Mix':
            self.real_A_edge, self.rec_A_edge, self.loss_Cycle_A = self.criterionEdge(self.real_A, self.rec_A)
            self.loss_Cycle_B = self.criterionCycle(self.real_B, self.rec_B)
            self.loss_Cycle_A = self.loss_Cycle_A * self.opt.lambda_cycle 
            self.loss_Cycle_B = self.loss_Cycle_B * self.opt.lambda_cycle

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_Cycle_A + self.loss_Cycle_B
        self.loss_G.backward()


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        Cycle_A = self.loss_Cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        Cycle_B = self.loss_Cycle_B.item()
        return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cycle_A', Cycle_A), ('D_B', D_B), ('G_B', G_B), ('Cycle_B', Cycle_B)])
        

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)

        latent_mask_list = []
        if self.opt.cycle_loss == 'Edge':
            real_A_edge = util.atten2im(self.real_A_edge.data)
            rec_A_edge = util.atten2im(self.rec_A_edge.data)
            real_B_edge = util.atten2im(self.real_B_edge.data)
            rec_B_edge = util.atten2im(self.rec_B_edge.data)
            latent_mask_list.extend([('real_A_edge', real_A_edge), ('rec_A_edge', rec_A_edge), ('real_B_edge', real_B_edge), ('rec_B_edge', rec_B_edge)])
        if self.opt.cycle_loss == 'Mix':
            real_A_edge = util.atten2im(self.real_A_edge.data)
            rec_A_edge = util.atten2im(self.rec_A_edge.data)
            latent_mask_list.extend([('real_A_edge', real_A_edge), ('rec_A_edge', rec_A_edge)])

        if self.opt.n_mask > 0:
            for i in range(0, self.latent_real_A.shape[1], 3):
                latent_mask_list.append((f'latent_real_A{int(i/3)+1}', util.tensor2im(self.latent_real_A.data[:, i:i+3])))
                latent_mask_list.append((f'mask_A{int(i/3)+1}', util.atten2im(self.mask_A.data[:, int(i/3):int(i/3)+1])))
            for i in range(0, self.latent_real_B.shape[1], 3):
                latent_mask_list.append((f'latent_real_B{int(i/3)+1}_{id}', util.tensor2im(self.latent_real_B.data[:, i:i+3])))
                latent_mask_list.append((f'mask_B{int(i/3)+1}_{id}', util.atten2im(self.mask_B.data[:, int(i/3):int(i/3)+1])))
        
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)] + latent_mask_list)


    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
