# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:59:21 2019

@author: hyw
"""
import torch
import torch.nn as nn


class Vnet1(nn.Module):

    def block_1_layer(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                )
        return block

    def block_2_layer(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),

                nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                )
        return block

    def block_3_layer(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),

                nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),

                nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                )
        return block

    def skip_block(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, 1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                )
        return block

    def down_block(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 2, stride=2),
                )
        return block

    def up_block_double(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                )
        return block

    def up_block_double(self, in_ch, out_ch):
        block = nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                )
        return block

    def dropout_block(self, x):
        block = nn.Sequential(
                nn.Dropout(p=x),
                )
        return block

    def __init__(self, start_ch = 12, in_ch = 1, out_ch = 2, dropout_p=0.3):
        super(Vnet1, self).__init__()

        self.block1 = self.block_1_layer(in_ch, start_ch)
        self.block1_skip = self.skip_block(in_ch, start_ch)
        self.block1_down = self.down_block(start_ch, start_ch*2)

        self.block2 = self.block_2_layer(start_ch*2, start_ch*2)
        self.block2_skip = self.skip_block(start_ch*2, start_ch*2)
        self.block2_down = self.down_block(start_ch*2, start_ch*4)
        
        self.block3 = self.block_3_layer(start_ch*4, start_ch*4)
        self.block3_skip = self.skip_block(start_ch*4, start_ch*4)
        self.block3_down = self.down_block(start_ch*4, start_ch*8)
        
        self.block4 = self.block_3_layer(start_ch*8, start_ch*8)
        self.block4_skip = self.skip_block(start_ch*8, start_ch*8)
        self.block4_down = self.down_block(start_ch*8, start_ch*16)
        
        self.block5_dropout = self.dropout_block(dropout_p)
        self.block5 = self.block_3_layer(start_ch*16, start_ch*16)
        self.block5_skip = self.skip_block(start_ch*16,start_ch*16)
        self.block5_up = self.up_block_double(start_ch*16,start_ch*8)
        
        self.block6 = self.block_3_layer(start_ch*16, start_ch*8)
        self.block6_skip = self.skip_block(start_ch*16, start_ch*8)
        self.block6_up = self.up_block_double(start_ch*8,start_ch*4)
        
        self.block7 = self.block_3_layer(start_ch*8, start_ch*4)
        self.block7_skip = self.skip_block(start_ch*8, start_ch*4)
        self.block7_up = self.up_block_double(start_ch*4,start_ch*2)
        
        self.block8 = self.block_2_layer(start_ch*4, start_ch*2)
        self.block8_skip = self.skip_block(start_ch*4, start_ch*2)
        self.block8_up = self.up_block_double(start_ch*2,start_ch)

        self.block9 = self.block_1_layer(start_ch*2, start_ch)
        self.block9_skip = self.skip_block(start_ch*2, start_ch)

        self.deep1 = self.up_block_double(start_ch*8, start_ch*4)

        self.deep2 = self.up_block_double(start_ch*8, start_ch*4)

        self.deep3 = self.up_block_double(start_ch*6, start_ch*3)

        self.softmax = nn.Sequential(
                #nn.Conv3d(start_ch, out_ch, 1, 1),
                nn.Conv3d(start_ch*4, out_ch, 1, 1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                
                nn.Softmax(dim=1)
                )
        
    def forward(self, inputs):
        
        block1_out = self.block1(inputs) + self.block1_skip(inputs)
        block2_input = self.block1_down(block1_out)
        
        block2_out = self.block2(block2_input) + self.block2_skip(block2_input)
        block3_input = self.block2_down(block2_out)
        
        block3_out = self.block3(block3_input) + self.block3_skip(block3_input)
        block4_input = self.block3_down(block3_out)
        
        block4_out = self.block4(block4_input) + self.block4_skip(block4_input)
        block5_input = self.block4_down(block4_out)
        
        block5_dropout = self.block5_dropout(block5_input)
        block5_out = self.block5(block5_dropout) + self.block5_skip(block5_dropout)
        block6_input_before = self.block5_up(block5_out)
        
        block6_input = torch.cat([block6_input_before, block4_out], dim=1)
        block6_out = self.block6(block6_input) + self.block6_skip(block6_input)
        block7_input_before = self.block6_up(block6_out)
        
        block7_input = torch.cat([block7_input_before, block3_out], dim=1)
        block7_out = self.block7(block7_input) + self.block7_skip(block7_input)
        block8_input_before = self.block7_up(block7_out)
        
        block8_input = torch.cat([block8_input_before, block2_out], dim=1)
        block8_out = self.block8(block8_input) + self.block8_skip(block8_input)
        block9_input_before = self.block8_up(block8_out)
        
        block9_input = torch.cat([block9_input_before, block1_out], dim=1)
        block9_out = self.block9(block9_input) + self.block9_skip(block9_input)
        
        deep1_out = self.deep1(block6_input_before)
        
        deep2_input = torch.cat([block7_input_before, deep1_out], dim=1)
        deep2_out = self.deep2(deep2_input)
        
        deep3_input = torch.cat([block8_input_before, deep2_out], dim=1)
        deep3_out = self.deep3(deep3_input)
        
        final_input = torch.cat([block9_out, deep3_out], dim=1)
        
        # final_input = block9_out
        output = self.softmax(final_input)
        
        return output


if __name__ == '__main__':

    net = Vnet1(start_ch=16, out_ch=23)

    net = net.cuda(0)
    data = torch.randn((1, 1, 80, 160, 160)).cuda()
    res = net(data)
    print(res.size())