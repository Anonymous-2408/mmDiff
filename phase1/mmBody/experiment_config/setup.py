import argparse
import logging
import os

 
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    # dataset config
    parser.add_argument('--input_channel', '-ic', type=int, default=25, help='Number of Input Channel')
    parser.add_argument('--classes', '-c', type=int, default=17, help='Number of output classes')
    parser.add_argument('--test_scene', '-ts', type=str, default="all", help='Number of output classes')
    # modalities = ["Depth", "RadarP", "RGB", "RadarImg5"]
    modalities = ["Depth","RadarP4"]
    # modalities = ["RadarP4","Depth", "RadarP", "RadarImg5"]

    # training
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    parser.add_argument('--l1norm', '-l1', type=str, default=False, help='Training with L1 regularization.')
    parser.add_argument('--l2norm', '-l2', type=str, default=False, help='Training with L2 regularization.')
    parser.add_argument('--val_epoch', '-ve', metavar='VE', type=int, default=39892, help='Number of epochs for validation.') 

    # model
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling') # currently only False implemented
    parser.add_argument('--pretrain_main', '-pm', type=str, default=False, help='Load model from a .pth file') # whether use pretrained fusion model
    parser.add_argument('--pretrain_encoder', '-pe', type=str, default=False, help='Load model from a .pth file') # whether use pretrained encoder model
    parser.add_argument('--pretrain_depth', '-pd', type=str, default=False, help='Load model from a .pth file') # whether use pretrained depth model
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file') # currently only False implemented
    parser.add_argument('--joint_num', '-jn', type=int, default=17,
                        help='Number of joints')  
    parser.add_argument('--night', '-night', type=str, default=False,
                        help='Overnight Training')  # currently only False implemented
    
    # training
    
    parser.add_argument('--mode', '-mode', type=str, default="pretrain_depth",
                        help='Training Mode')  # "base", "pretrain_depthfusion", "base-attention", "pretrain_depth", (attention-aggr)
    parser.add_argument('--depthfile', '-depth', type=str, default="best_model_depth_full_drop19.pth",
                        help='Training Mode')  # "base", "pretrain_depthfusion", "base-attention", "pretrain_depth"
    
    


    return parser.parse_args(), modalities