default_scope = 'cmae'
import os

# model settings
model = dict(
    type='CMAE',
    backbone=dict(
        online=dict(type='CMAEViT', arch='b', patch_size=16, mask_ratio=0.65),
        target=dict(type='CMAEViT', arch='b', patch_size=16, mask_ratio=0.0,final_norm=False),
    ),
    neck=dict(
        pixel=dict(
            type='MAEPretrainDecoder',
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
        ),
        feature=dict(
            type='MAEPretrainDecoder',
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_depth=2,
            decoder_num_heads=16,
            mlp_ratio=4.,
        ),
        projector=dict(
            type='NonLinearNeck',
            in_channels=768,
            hid_channels=1536,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
    ),

    head=dict(type='CMAEPretrainHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=1536,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        temperature=0.07,
        norm_pix=True,
        patch_size=16,
        ct_weight=1.0,
        rc_weight=1.0)
)

# dataset settings
dataset_type = 'CMAEDataset'
data_root = os.path.join(os.getenv("INPUT_PATH", './'), 'data/ImageNet/')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type="LoadImageNetFromFile", ),
    dict(type='RandomResizedCrop', scale=256, crop_ratio_range=(0.2, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img',]),
    dict(type='ToTensor', keys=['img',]),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_ann="ImageNet_train.json",
        pipeline=train_pipeline,
        pixel=31,
    ),
)

# optimizer
update_interval=1
gpu_num = 8*4
base_lr = 1.5e-4
lr = base_lr * train_dataloader['batch_size'] * update_interval * gpu_num / 256

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=1560,
        by_epoch=True,
        begin=40,
        end=1600,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1600)

custom_hooks = [dict(type='MomentumUpdateHook',end_momentum=0.996)]

# checkpoint saving
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])
#
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')

# custom_hooks = [dict(type='SelfSupVisualizationHook', interval=1)]

log_level = 'INFO'
load_from = None
compile_options = dict(backend='inductor', mode='max-autotune')
cfg = dict(compile=compile_options)
resume = True
randomness = dict(seed=60, diff_rank_seed=True)


