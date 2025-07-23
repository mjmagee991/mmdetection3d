_base_ = 'second_hv_secfpn_8xb6-amp-80e_kitti-3d-car.py'

voxel_size = [0.1, 0.1, 0.1]
point_cloud_range = [-80, -80, -3, 80, 80, 1]

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    middle_encoder=dict(sparse_shape=[41, 1600, 1600]),
    bbox_head=dict(
        anchor_generator=dict(
            ranges=[[-80.0, -80.0, -1.78, 80, 80.0, -1.78]])))

data_root = 'data/kitti/'
class_names = ['Car']
backend_args=None
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline)))
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline))
