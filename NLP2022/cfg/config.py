from yacs.config import CfgNode as CN

def get_default_config():
    _C = CN()

    # DATA
    _C.DATA = CN()
    _C.DATA.dataset_name = "CUB_200_2011"
    _C.DATA.data_dir = "/home/hong/datasets"
    _C.DATA.img_size = 256
    _C.DATA.img_channels = 3
    _C.DATA.num_captions = 10
    _C.DATA.num_words = 18

    # METRICS
    _C.METRICS = CN()
    _C.METRICS.eval_backbone = "InceptionV3_tf"
    _C.METRICS.ref_dataset = "test"
    
    # MODEL
    _C.MODEL = CN()
    _C.MODEL.z_dim = 100
    _C.MODEL.c_dim = 256
    _C.MODEL.channel_base = 32

    # TRAINING
    _C.TRAINING = CN()
    _C.TRAINING.device = None
    _C.TRAINING.batch_size = 32
    _C.TRAINING.num_workers = None
    _C.TRAINING.G_lr = 0.0001
    _C.TRAINING.D_lr = 0.0004
    _C.TRAINING.num_epochs = 1000
    
    return _C.clone()