default_scope = 'cmae'
model = dict(
    type='Classification',
    backbone=dict(
        type='MIMVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint='work_dirs/init_f/cmae_bs4096_ep1600.pth'),
        arch='b',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=False),
    head=dict(
        type='MAEFinetuneHead',
        num_classes=1000,
        embed_dim=768,
        label_smooth_val=0.1),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))


# dataset settings
dataset_type = 'ImageNetDataset'
data_root =  './data/ImageNet/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type="LoadImageNetFromFile", ),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[124,116,104], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[  123.675,116.28,103.53],
        fill_std=[ 58.395,57.12,57.375]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img','label' ]),
    dict(type='ToTensor', keys=['img', 'label']),
]
test_pipeline = [
    dict(type="LoadImageNetFromFile", ),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'label']),
    dict(type='ToTensor', keys=['img', 'label']),
]

# prefetch
prefetch = False


# dataset summary
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_ann="ImageNet_train.json",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_ann="ImageNet_val.json",
        pipeline=test_pipeline,)
)
test_dataloader = val_dataloader
evaluation = dict(interval=1, topk=(1, 5))


base_lr = 2.5e-4
gpu_num=8*1
lr = base_lr * train_dataloader['batch_size']  * gpu_num / 256.
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=lr, weight_decay=0.05, betas=(0.9, 0.999),model_type='vit'),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.65,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_begin=0,
    val_interval=10)
val_cfg = dict()
test_cfg = dict()

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator


default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=4),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = []

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])
#


log_level = 'INFO'
load_from = None
compile_options = dict(backend='inductor', mode='max-autotune')
cfg = dict(compile=compile_options)
resume = True
randomness = dict(seed=60, diff_rank_seed=True)


