import torch
import torch.nn as nn
from .segformer import *
from typing import Tuple
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x


class SegU_decoder(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

        

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims,dims*4)
        self.mixffn2 = MixFFN_skip(dims*2,dims*8)
        self.mixffn3 = MixFFN_skip(dims*5,dims*20)
        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        tem1 = tx[:,:3136,:].reshape(B, -1, C) 
        tem2 = tx[:,3136:4704,:].reshape(B, -1, C*2)
        tem3 = tx[:,4704:5684,:].reshape(B, -1, C*5)
        tem4 = tx[:,5684:6076,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2


class BridgeLayer_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        # self.mixffn1 = MixFFN(dims,dims*4)
        self.mixffn2 = MixFFN(dims*2,dims*8)
        self.mixffn3 = MixFFN(dims*5,dims*20)
        self.mixffn4 = MixFFN(dims*8,dims*32)
        
        
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        # tem1 = tx[:,:3136,:].reshape(B, -1, C) 
        tem2 = tx[:,:1568,:].reshape(B, -1, C*2)
        tem3 = tx[:,1568:2548,:].reshape(B, -1, C*5)
        tem4 = tx[:,2548:2940,:].reshape(B, -1, C*8)

        # m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2



class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        outs = []

        sk1 = bridge4[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2) 
        sk2 = bridge4[:,3136:4704,:].reshape(B, 28, 28, C*2).permute(0,3,1,2) 
        sk3 = bridge4[:,4704:5684,:].reshape(B, 14, 14, C*5).permute(0,3,1,2) 
        sk4 = bridge4[:,5684:6076,:].reshape(B, 7, 7, C*8).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class BridegeBlock_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if (type(x) == list):
            # print("-----1-----")
            outs.append(x[0])
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        

        # sk1 = bridge2[:,:3136,:].reshape(B, 56, 56, C).permute(0,3,1,2) 
        sk2 = bridge4[:,:1568,:].reshape(B, 28, 28, C*2).permute(0,3,1,2) 
        sk3 = bridge4[:,1568:2548,:].reshape(B, 14, 14, C*5).permute(0,3,1,2) 
        sk4 = bridge4[:,2548:2940,:].reshape(B, 7, 7, C*8).permute(0,3,1,2) 

        # outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios,token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class MISSFormer(nn.Module):
    def __init__(self, num_classes=9, in_ch=3, token_mlp_mode="mix_skip", encoder_pretrained=True):
        super().__init__()
    
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(224, dims, layers,in_ch, token_mlp_mode)

        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = BridegeBlock_4(64, 1, self.reduction_ratios)

        self.decoder_3= MyDecoderLayer((d_base_feat_size,d_base_feat_size), in_out_chan[3], heads[3], reduction_ratios[3],token_mlp_mode, n_class=num_classes)
        self.decoder_2= MyDecoderLayer((d_base_feat_size*2,d_base_feat_size*2),in_out_chan[2], heads[2], reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1= MyDecoderLayer((d_base_feat_size*4,d_base_feat_size*4), in_out_chan[1], heads[1], reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0= MyDecoderLayer((d_base_feat_size*8,d_base_feat_size*8), in_out_chan[0], heads[0], reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

        
    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder) #list

        b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------     
        # print("stage3-----")   
        tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")   
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")   
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")  
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        return tmp_0

         
