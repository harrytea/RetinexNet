import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(), nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet  = DecomNet()
        self.RelightNet= RelightNet()

    def forward(self, input_low, input_high):
        # Forward DecompNet
        R_low, I_low   = self.DecomNet(input_low)
        # R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        # I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        # Compute losses
        # self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low)
        # self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        # self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low)
        # self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        # self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
        # self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        # self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        # self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        # self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        # self.loss_Decom = self.recon_loss_low + \
        #                   self.recon_loss_high + \
        #                   0.001 * self.recon_loss_mutal_low + \
        #                   0.001 * self.recon_loss_mutal_high + \
        #                   0.1 * self.Ismooth_loss_low + \
        #                   0.1 * self.Ismooth_loss_high + \
        #                   0.01 * self.equal_R_loss
        # self.loss_Relight = self.relight_loss + \
        #                     3 * self.Ismooth_loss_delta

        # Decom output
        self.output_R_low    = R_low.detach().cpu()
        self.output_I_low    = I_low_3.detach().cpu()
        # self.output_R_high   = R_high.detach().cpu()
        # self.output_I_high   = I_high_3.detach().cpu()

        self.output_I_delta  = I_delta_3.detach().cpu()
        self.output_S        = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))


    def evaluate(self, epoch_num, val_loader, train_phase, wandb):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        with torch.no_grad():
            for _, (high, low) in enumerate(val_loader, 0):
                high, low = high.cuda(), low.cuda()
                if train_phase == "Decom":
                    self.forward(low, high)
                    R_low,  I_low  = self.output_R_low[0],  self.output_I_low[0]
                    R_high, I_high = self.output_R_high[0], self.output_I_high[0]
                    low, high      = low[0].detach().cpu(), high[0].detach().cpu()
                    Rec_l, Rec_h   = R_low*I_low, R_high*I_high

                    low_image      = np.concatenate([low, R_low, I_low, Rec_l], axis=2)
                    high_image     = np.concatenate([high, R_high, I_high, Rec_h], axis=2)

                    '''  log images  '''
                    low_image  = np.transpose(low_image, (1, 2, 0))
                    high_image = np.transpose(high_image, (1, 2, 0))
                    low_image  = Image.fromarray(np.clip(low_image * 255.0, 0, 255.0).astype('uint8'))
                    high_image = Image.fromarray(np.clip(high_image * 255.0, 0, 255.0).astype('uint8'))
                    L_Img = wandb.Image(low_image, caption="(epoch {})LtoR: Ori, R, I, Rec".format(epoch_num))
                    wandb.log({"Decom low image": L_Img, "epoch": epoch_num})
                    H_Img = wandb.Image(high_image, caption="(epoch {})LtoR: Ori, R, I, Rec".format(epoch_num))
                    wandb.log({"Decom high image": H_Img, "epoch": epoch_num})


                if train_phase == "Relight":
                    self.forward(low, high)
                    R_low,  I_low  = self.output_R_low[0],  self.output_I_low[0]
                    R_high, I_high = self.output_R_high[0], self.output_I_high[0]
                    low, high      = low[0].detach().cpu(), high[0].detach().cpu()
                    Rec_l, Rec_h   = R_low*I_low, R_high*I_high
                    I_delta, out   = self.output_I_delta[0], self.output_S[0]

                    low_image      = np.concatenate([low, R_low, I_low, Rec_l], axis=2)
                    high_image     = np.concatenate([high, R_high, I_high, Rec_h], axis=2)
                    low2high       = np.concatenate([high, R_low, I_low, I_delta, out], axis=2)

                    '''  log images  '''
                    low_image  = np.transpose(low_image, (1, 2, 0))
                    high_image = np.transpose(high_image, (1, 2, 0))
                    low2high   = np.transpose(low2high, (1, 2, 0))

                    low_image  = Image.fromarray(np.clip(low_image * 255.0, 0, 255.0).astype('uint8'))
                    high_image = Image.fromarray(np.clip(high_image * 255.0, 0, 255.0).astype('uint8'))
                    low2high   = Image.fromarray(np.clip(low2high * 255.0, 0, 255.0).astype('uint8'))

                    L_Img      = wandb.Image(low_image, caption="(epoch {})LtoR: Ori, R, I, Rec".format(epoch_num))
                    wandb.log({"Relight low image": L_Img, "epoch": epoch_num})
                    H_Img      = wandb.Image(high_image, caption="(epoch {})LtoR: Ori, R, I, Rec".format(epoch_num))
                    wandb.log({"Relight high image": H_Img, "epoch": epoch_num})
                    LtoH       = wandb.Image(low2high, caption="(epoch {})LtoR: GT, lowR, lowI, RecI, Rec".format(epoch_num))
                    wandb.log({"Reconstruct image": LtoH, "epoch": epoch_num})


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)


    def train_one(self,train_loader, val_loader, epoch, lr, ckpt_dir, eval_every_epoch, train_phase, rank, wandb):

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(), lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(), lr=lr[0], betas=(0.9, 0.999))

        self.train_phase= train_phase

        start_epoch = 0
        if rank==0:
            print("Start training for phase %s, with start epoch %d : " % (self.train_phase, start_epoch))

        start_time = time.time()
        for epoch in range(start_epoch, epoch):
            loss_epoch  = 0
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr

            for _, (high, low) in enumerate(train_loader, 0):
                # Feed-Forward to the network and obtain loss
                high, low = high.cuda(), low.cuda()
                self.forward(low,  high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                    loss_epoch += loss
                    if rank==0:
                        wandb.log({"loss_Decom": loss, "epoch": epoch})
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()
                    loss_epoch += loss
                    if rank==0:
                        wandb.log({"loss_Relight": loss, "epoch": epoch})

            if rank==0:
                print("%s Epoch: [%2d] time: %4.4f, loss: %.6f" % (train_phase, epoch + 1, time.time()-start_time, loss_epoch))

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1)%eval_every_epoch==0 and rank==0:
                self.evaluate(epoch + 1, val_loader, train_phase=train_phase, wandb=wandb)
                self.save(epoch, ckpt_dir)
        if rank==0:
            print("Finished training for phase %s." % train_phase)



    def predict(self, wandb, test_loader):
        print("Testing samples %d" % len(test_loader))
        with torch.no_grad():
            for _, (img, name) in enumerate(tqdm(test_loader), 0):
                img = img.cuda()
                self.forward(img, img)
                R_low,  I_low  = self.output_R_low[0],  self.output_I_low[0]
                img            = img[0].detach().cpu()
                Rec_l          = R_low*I_low
                I_delta, out   = self.output_I_delta[0], self.output_S[0]

                low2high       = np.concatenate([img, R_low, I_low, Rec_l, I_delta, out], axis=2)

                '''  log images  '''
                low2high   = np.transpose(low2high, (1, 2, 0))
                low2high   = Image.fromarray(np.clip(low2high * 255.0, 0, 255.0).astype('uint8'))
                LtoH       = wandb.Image(low2high, caption=name)
                wandb.log({"LtoR: Low, lowR, lowI, RecL, RecI, Rec": LtoH})
                # wandb.log({"LtoR: high, highR, highI, RecH, RecI, Rec": LtoH})
                torch.cuda.empty_cache()

