_base_ = 'pv_rcnn_8xb2-80e_kitti-3d-3class.py'

voxel_size = [0.4, 0.4, 0.1]
point_cloud_range = [-60, -60, -3, 60, 60, 1]

dataset_type = 'KittiDataset'
input_modality = dict(use_lidar=True, use_camera=False)
data_root = 'data/kitti/'
class_names = ['Car']
metainfo = dict(CLASSES=class_names)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5)),
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
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointSample', num_points=16000),
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

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    middle_encoder=dict(sparse_shape=[41, 300, 300]),
    points_encoder=dict(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(
                type='StackedSAModuleMSG',
                in_channels=16,
                scale_factor=1,
                radius=(0.4, 0.8),
                sample_nums=(8, 8),
                mlp_channels=((16, 16), (16, 16)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=32,
                scale_factor=2,
                radius=(0.8, 1.2),
                sample_nums=(8, 16),
                mlp_channels=((32, 32), (32, 32)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=4,
                radius=(1.2, 2.4),
                sample_nums=(8, 16),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=8,
                radius=(2.4, 4.8),
                sample_nums=(8, 16),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True)
        ]),
    rpn_head=dict(num_classes=1),
    roi_head=dict(
        num_classes=1,
        bbox_roi_extractor=dict(grid_size=4),
        bbox_head=dict(num_classes=1)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(  # for Car
                    _delete_=True,
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)),
        rcnn=dict(
            assigner=dict(  # for Car
                    _delete_=True,
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1))))

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            metainfo=metainfo)))
test_dataloader = dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne'),
            pipeline=test_pipeline,
            metainfo=metainfo))
eval_dataloader = dict(
        dataset=dict(
            data_prefix=dict(pts='training/velodyne'),
            pipeline=test_pipeline,
            metainfo=metainfo))
