import os
import argparse
import sys
from collections import defaultdict, deque
import pickle

import numpy as np
from PIL import Image
import cv2

from sahi.utils.coco import Coco
from sahi.utils.cv import get_bool_mask_from_coco_segmentation

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss

from predict_utils import show_mask, show_points, calculate_metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Add the SAM directory to the system path
sys.path.append("./")
from SAM import sam_model_registry
from SAM.utils.transforms import ResizeLongestSide
from dataset import custom_text, augmentation, custom_text_test, custom_text_test_one, custom_text_test_realdata
import time

# NUM_GPUS = torch.cuda.device_count()
NUM_GPUS = 1
DEVICE = 'cuda'
NUM_WORKERS = 0  # https://github.com/pytorch/pytorch/issues/42518


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class SAMFinetuner(pl.LightningModule):

    def __init__(
            self,
            model_type,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10,
            args=None
        ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for k, param in self.model.prompt_encoder.named_parameters():
                if 'modulate_prompt' in k:
                  param.requires_grad = True
                else:
                  param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        # for k, v in self.model.prompt_encoder.named_parameters():
        #     print(k, v.requires_grad)
                
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.val_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.metrics_interval = metrics_interval
        
        self.transform = ResizeLongestSide(args.image_size)
        self.args = args
        self.mask_threshold : float = 0.0
        self.validation_step_outputs = []
        self.save_base = args.output_dir + '/val_results'
        self.metrics_interval_num = 0
        self.miou_activation_map = np.zeros((1024,1024,1), dtype=float)
        self.img = np.zeros((1024,1024,3), dtype=float)

    def forward(self, batch):
        imgs = batch[0] 
        gt_masks = batch[1]
        points = batch[2]
        point_labels = batch[3]
        _, _, H, W = imgs.shape
        
        features = self.model.image_encoder(imgs)
        ###
        num_masks = imgs.shape[0]

        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        confidence_predictions = []
        tp, fp, fn, tn = [], [], [], []
        for feature, point, point_label, gt_mask in zip(features, points, point_labels, gt_masks):
            
            point_coords = self.transform.apply_coords_torch(point, (self.args.image_size, self.args.image_size))
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_label, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
            point = (coords_torch, labels_torch)
            
            # Embed prompts
            # sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            #     points=point,
            #     boxes=None,
            #     masks=None,
            # )
            
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                feature=feature,
                points=point,
                boxes=None,
                masks=None,
            )
            
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            
            predictions.append(masks)
            confidence_predictions.append(iou_predictions)
            # masks = masks > self.mask_threshold
            
            # Compute the iou between the predicted masks and the ground truth masks
            
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                gt_mask.unsqueeze(0).unsqueeze(0),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            # Compute the loss
            
            masks = masks.squeeze(1).flatten(1)
            gt_mask = gt_mask.unsqueeze(0).flatten(1)
            
            loss_focal += sigmoid_focal_loss(masks, gt_mask.float(), num_masks)
            loss_dice += dice_loss(masks, gt_mask.float(), num_masks)
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'confidence_predictions': confidence_predictions,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }
    
    def training_step(self, batch, batch_nb):
        # imgs, bboxes, labels = batch
        outputs = self(batch)
        # outputs = self(imgs, bboxes, labels)

        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])

        # aggregate step metics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        return metrics
    
    def validation_step(self, batch, batch_nb):
        # imgs, bboxes, labels = batch
        # outputs = self(imgs, bboxes, labels)
        outputs = self(batch)
        
        imgs = batch[0] 
        gt_masks = batch[1]
        points = batch[2]
        point_labels = batch[3]
        pred_masks = outputs['predictions']
        outputs.pop("confidence_predictions")
        outputs.pop("predictions")
        # num = self.metrics_interval_num
        # for i, (img, gt_mask, pred_mask, point, point_label) in enumerate(zip(imgs, gt_masks, pred_masks, points, point_labels)):
        #     img = img.permute(1,2,0).detach().cpu().numpy()
        #     img = augmentation.DeNormalize(img).astype(int)
            
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(img)
        #     # ##mask save
        #     # pred_mask = pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(int)
        #     # pred_mask = pred_mask.detach().cpu().numpy().astype(int)
        #     pred_mask = pred_mask > 0.0

        #     # # iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
        #     show_mask(pred_mask, plt.gca())
        #     show_points(point.detach().cpu().numpy(), point_label.detach().cpu().numpy(), plt.gca())
            
        #     # # plt.title(f"Mask {i+1}, IOU: {iou:.3f}, F-score: {f_score:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, Confidence: {score:.3f}", fontsize=12)
        #     # # plt.axis('off')
        #     filename = f"{num}_pred_{i}.png"
        #     plt.savefig(os.path.join(self.save_base, filename), bbox_inches='tight', pad_inches=0)
        #     plt.close()
            
            
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(img)
        #     ##mask save
        #     gt_mask = gt_mask.detach().cpu().numpy()
            
        #     # iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
        #     show_mask(gt_mask, plt.gca())
        #     show_points(point.detach().cpu().numpy(), point_label.detach().cpu().numpy(), plt.gca())
            
        #     filename = f"{num}_gt_{i}.png"
        #     plt.savefig(os.path.join(self.save_base, filename), bbox_inches='tight', pad_inches=0)
        #     plt.close()
        
        # self.metrics_interval_num += 1
        
        ## validation log 
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.val_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.val_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        self.validation_step_outputs.append(per_mask_iou)
        metrics = {"val_per_mask_iou": per_mask_iou}
        # self.log_dict(metrics)
        self.log("val_per_mask_iou", per_mask_iou, sync_dist=True)
        
        return metrics
    
    def on_validation_epoch_end(self):

        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average iou", epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            ## warmup change
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            # collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            # collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader
    
    def test_step(self, batch, batch_nb):
        # imgs, bboxes, labels = batch
        # outputs = self(imgs, bboxes, labels)
        outputs = self(batch)
        
        imgs = batch[0] 
        gt_masks = batch[1]
        points = batch[2]
        point_labels = batch[3]
        image_ids = batch[4]
        pred_masks = outputs['predictions']
        confidence_predictions = outputs['confidence_predictions']
        outputs.pop("predictions")
        outputs.pop("confidence_predictions")
        num = self.metrics_interval_num
        for i, (img, gt_mask, pred_mask, point, point_label, image_id, score) \
        in enumerate(zip(imgs, gt_masks, pred_masks, points, point_labels, image_ids, confidence_predictions)):
            img = img.permute(1,2,0).detach().cpu().numpy()
            img = augmentation.DeNormalize(img).astype(int)
            score = score.detach().cpu().numpy()[0][0]
            # plt.figure(figsize=(10,10))
            # plt.imshow(img)
            id = image_id.split('.')[0]
            pred_mask = pred_mask > 0.0
            
            g_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            iou, f_score, precision, recall = calculate_metrics(pred_mask*1, g_mask*1)
            # show_mask(pred_mask, plt.gca())
            new_p = point.int()[0].detach().cpu()
            point = point.detach().cpu().numpy()
            self.miou_activation_map[int(new_p[1].item()), int(new_p[0].item()), :] = iou.item()
            # show_points(point, point_label.detach().cpu().numpy(), plt.gca())
            # plt.title(f"Mask, IOU: {iou:.3f}, F-score: {f_score:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, Confidence: {score:.3f}", fontsize=12)
            # plt.axis('off')
            # filename = f"{new_p[0].item(), new_p[1].item()}_pred.png"
            # filename = f"finetuned_{id, int(new_p[1].item()), int(new_p[0].item())}_pred.png"
            # plt.savefig(os.path.join(self.save_base, filename), bbox_inches='tight', pad_inches=0)
            # plt.close()
        
            # plt.figure(figsize=(10,10))
            # plt.imshow(img)
            #     # ##mask save
            # gt_mask = gt_mask.detach().cpu().numpy()
            
            # # iou, f_score, precision, recall = calculate_metrics(pred_mask, gt_mask)
            # show_mask(gt_mask, plt.gca())
            # show_points(point, point_label.detach().cpu().numpy(), plt.gca())
            
            # filename = f"{id}_gt.png"
            # plt.axis('off')
            # plt.savefig(os.path.join(self.save_base, filename), bbox_inches='tight', pad_inches=0)
            # plt.close()
            
        # img = img.permute(1,2,0).detach().cpu().numpy()
        # img = augmentation.DeNormalize(img).astype(int)
        self.img = img
        ## validation log 
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.val_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.val_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        self.validation_step_outputs.append(per_mask_iou)
        metrics = {"val_per_mask_iou": per_mask_iou}
        # self.log_dict(metrics)
        self.log("val_per_mask_iou", per_mask_iou, sync_dist=True)
        
        return metrics

    def on_test_epoch_end(self):
        print('test_end')
        # plt.figure(figsize=(10,10))
        # print(self.img.shape, self.miou_activation_map.shape)
        # print(self.img.shape, self.miou_activation_map.shape)
        img = cv2.resize(self.img.astype('float32'), dsize=(1440,1080), interpolation=cv2.INTER_LINEAR).astype('int32')
        # ## interpolation
        # act_map = cv2.resize(self.miou_activation_map,
        #                             dsize=(512,512),
        #                             interpolation=cv2.INTER_LINEAR)
        act_map = cv2.resize(self.miou_activation_map, dsize=(1440,1080), interpolation=cv2.INTER_LINEAR)
        vmax = 0
        vmin = 1
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colormapping = cm.ScalarMappable(norm=norm, cmap='jet')
        plt.imshow((act_map * 255).astype(np.uint8), cmap='jet', vmin=0, vmax=1)
        plt.imshow(img, alpha=0.5)
        filename = f"activation_map_gt.png"
        plt.axis('off')
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        # plt.jet()
        cbar = plt.colorbar(colormapping, ax=ax, cax=cax) ## 컬러바 삽입
        # cbar = plt.colorbar(ax=ax, cax=cax) ## 컬러바 삽입
        
        plt.savefig(os.path.join(self.save_base, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average iou", epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to the data root")
    parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument("--freeze_image_encoder", action="store_true", help="freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", help="freeze prompt encoder")
    parser.add_argument("--freeze_mask_decoder", action="store_true", help="freeze mask decoder")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--steps", type=int, default=1500, help="number of steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=50, help="interval for logging metrics")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save the model")
    parser.add_argument("--test_only", action="store_true", help="test_only")

    args = parser.parse_args()

    
    train_dataset = custom_text.CustomText(
        data_root='../SAM_customizing/data/Custom_data',
        is_training=True,
        load_memory=False,
        cfg=args,
        transform= augmentation.Augmentation(is_training = True, size=args.image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    
    val_dataset = custom_text.CustomText(
        # data_root='./data/validation' if not args.random_gen else '../SAM_customizing/data/Custom_data' ,
        data_root = '../SAM_customizing/data/Custom_data' ,
        is_training=False,
        load_memory=False,
        cfg=args,
        transform= augmentation.Augmentation(is_training = False, size=args.image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    
    # load the dataset
    if args.test_only:
        test_dataset = custom_text_test.CustomText_test(
        # test_dataset = custom_text_test_one.CustomText_test_one(
            data_root='./data/validation',
            # data_root='./data/small',
            label_root='./polygon.txt',
            is_training=False,
            load_memory=False,
            cfg=args,
            transform= augmentation.Augmentation(is_training = False, size=args.image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        test_dataset = custom_text_test_one.CustomText_test_one(
        # test_dataset = custom_text_test_realdata.CustomText_test_realdata(
            # data_root='./data/validation',
            data_root='./data',
            label_root='./polygon.txt',
            is_training=False,
            load_memory=False,
            cfg=args,
            transform= augmentation.Augmentation(is_training = False, size=args.image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )

        val_dataset = test_dataset 
            
    # create the model
    model = SAMFinetuner(
        args.model_type,
        args.checkpoint_path,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder,
        freeze_mask_decoder=args.freeze_mask_decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metrics_interval=args.metrics_interval,
        args=args
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{step}-{val_per_mask_iou:.2f}', 
            save_last=True,
            save_top_k=1,
            monitor="val_per_mask_iou",
            mode="max",
            save_weights_only=True,
            every_n_train_steps=args.metrics_interval,
        ),
    ]
    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true' if NUM_GPUS > 1 else 'auto',
        # strategy='auto',
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=32,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=args.steps,
        val_check_interval=args.metrics_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0
    )
    if args.test_only:
        trainer.test(model, ckpt_path = './step=24198-val_per_mask_iou=0.98.ckpt', dataloaders=model.val_dataloader())
        # trainer.test(model, ckpt_path = None, dataloaders=model.val_dataloader())
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
