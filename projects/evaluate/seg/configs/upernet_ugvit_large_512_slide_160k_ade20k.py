_base_ = [
    '_base_/models/upernet_cae.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        _delete_=True,
        type='VUM',
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        n_cls_tokens=8,
        n_reg_tokens=4,
        in_chans=3,
        out_indices=[7, 11, 15, 23],
        drop_path_rate=0.15,
        frozen_stages=-1,
        use_checkpoint=False,
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=150,
        channels=1024,
    ),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=150
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

optimizer = dict(_delete_=True, type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=2)
find_unused_parameters = True

runner = dict(type='IterBasedRunnerAmp')
checkpoint_config = dict(by_epoch=False, interval=32000)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

