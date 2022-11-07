import torch
from torch import nn
from mmocr.models.textdet.detectors import FCENet
from mmdet.models.builder import build_head
from mmdet.models.builder import DETECTORS
import cv2
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import torch.nn.functional as F


@DETECTORS.register_module()
class DeepLabV3Plus(nn.Module):
    def __init__(self, nclass):
        super(DeepLabV3Plus, self).__init__()
        
        low_level_channels = 256
        high_level_channels = 2048
        
        self.head = ASPPModule(high_level_channels, (12, 24, 36))
        
        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))
        
        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
        
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        
        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)
    
    def forward(self, h, w, c1, c4):
        # print(c4.shape)
        c4 = self.head(c4)
        # print(c4.shape)
        
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        
        c1 = self.reduce(c1)
        
        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)
        
        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        
        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


@DETECTORS.register_module()
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))
    
    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates
        
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)
        
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))
    
    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


@DETECTORS.register_module()
class TPSNet(FCENet):
    
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 recog_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None,
                 from_p2=False):
        super(TPSNet, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, show_score, init_cfg)
        
        self.recog_head = build_head(recog_head)
        self.from_p2 = from_p2
        
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.seg_model = DeepLabV3Plus(nclass=2)
        self.count_ = 0
    
    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x, c1, c4 = self.extract_feat(img, is_get_c1_c4=True)  # x：对resnet 4个layer的FPN结果 c1,c4:resnet的第三第四个layer
        preds = self.bbox_head(x[1:])  # 计算FPN特征分别在两个分支后的卷积结果
        losses = self.bbox_head.loss(preds, **kwargs)
        
        seg_mask = kwargs['seg_mask']
        seg_mask = seg_mask.long()
        h, w = seg_mask.shape[1:]
        seg_pred = self.seg_model(h, w, c1, c4)
        seg_loss = self.seg_criterion(seg_pred, seg_mask)
        if self.count_ % 1000 == 0:
            seg_pred = torch.argmax(seg_pred, dim=1)
            cv2.imshow('target', np.array(ToPILImage()(seg_mask[0].cpu().type(torch.uint8))) * 255)
            cv2.imshow('pred', np.array(ToPILImage()(seg_pred[0].cpu().type(torch.uint8))) * 255)
            cv2.waitKey(0)
        self.count_ += 1
        
        losses['loss_seg'] = seg_loss
        # print(losses)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        # cv2.imshow('in', np.array(ToPILImage()(img[0].cpu())))
        x = self.extract_feat(img)
        outs = self.bbox_head(x[1:])
        
        # class_img = outs[0][0][0][2:, :, :].cpu()
        # class_img = torch.argmax(class_img, 0, keepdim=True).type(torch.uint8)
        # print(class_img.shape)
        # cv2.imshow('out', np.array(ToPILImage()(class_img * 255)))
        # cv2.waitKey(0)
        
        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs
        
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)), [img_metas[i]], rescale) for i in
                range(len(img_metas))
            ]
            print('len(img_metas) > 1')
            exit()
        else:
            boundaries = [
                self.bbox_head.get_boundary(outs, img_metas, rescale)
            ]
        
        boundaries = [self.recog_head.my_simple_test(x[:-1], boundaries[0], img_metas=img_metas, rescale=rescale)]
        
        return boundaries
