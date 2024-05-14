import copy
import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from torchvision import transforms, utils

from evaluate_p4trans import evaluate
from model import P4Transformer

from p4trans_scheduler import WarmupMultiStepLR


from radar_loader import MMPoseLoader
import gc


RECORD_FILE = "record_p4trans.txt"
CHECKPOINT_FILE = 'best_model_mars_test.pth'

def train_model(
        model,
        train_dataset,
        test_dataset,
        device,
        modalities,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        l1_flag = False,
        l1_lambda = 1e-9,
        l2_flag = False,
        l2_lambda = 1e-11,
        val_epochs = 2000,
        scene = None
):

    print("training start")

    # 2. Split into train / validation partitions
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    print("loader complete")


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
   
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0


    # # evaluation setup
    val_score, pck_acc, mean_pjpe, max_pjpe, pa_mpjpe = evaluate(model, val_loader, global_step, device, amp, batch_size,
                                                       test_flag=False, joint_num=args.joint_num, save_flag=f"")
    best_loss = val_score # a large number
    print(val_score, pck_acc, mean_pjpe, max_pjpe, pa_mpjpe)

    # return None


    print("train setup complete")



    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
       
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # print("start")
                selected_joints =  [0,1, 4, 7, 2, 5, 8, 6, 12, 15, 24,  16, 18,  20, 17, 19, 21]

                joints = batch["joints"][:,selected_joints,:].to(device=device, dtype=torch.float32)
                radar = batch["RadarP4"].to(device=device, dtype=torch.float32) # B, N, X

                
                

                # normalize joints with the center of body
                joints_original = copy.deepcopy(joints)
                joints_centers = joints_original[:,[0],:]
                joints = torch.sub(joints, joints_centers)


                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    joints_predict, _, _, _, _ = model(radar)

                    loss = 100* criterion(joints_predict, joints.float())*1.0                                                

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()


                    pbar.update(joints.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    pbar.set_postfix(**{'loss (batch)': loss.item()})


               # Evaluation round

                division_step = int(val_epochs/batch_size)

                if division_step > 0:
                    if global_step % division_step == 0:
                        print("start evaluation")


                        val_score, pck_acc,  mean_pjpe, max_pjpe = evaluate(model, val_loader, global_step, device, amp, batch_size, test_flag=False, joint_num=args.joint_num
                                                                            , l1_flag=l1_flag, l2_flag=l2_flag, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
                        train_loss = epoch_loss / division_step
                        epoch_loss = 0

                        scheduler.step(val_score)
                        print(optimizer.param_groups[0]['lr'])

                        record_file = open(RECORD_FILE, "a")
                        record_file.writelines(f"Division step: {global_step}\n")
                        record_file.writelines(f"epoch_loss: {train_loss}, val_score: {val_score}, pck_acc: {pck_acc}\n")
                        record_file.close()

                        logging.info('train_loss, val_loss, pck_acc, mean_pjpe, max_pjpe: {}, {}, {}, {}, {}'.format(train_loss, val_score, pck_acc, mean_pjpe, max_pjpe))



                        if val_score < best_loss:

                            best_loss = val_score

                            # save model weights
                            logging.info('Save model...')
                            savepath = os.path.join("checkpoints", CHECKPOINT_FILE)
                            logging.info('Saving at %s' % savepath)
                            state = {
                                'epoch': global_step,
                                'best_loss': val_score,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'grad_scaler_state_dict': grad_scaler.state_dict()
                            }
                            torch.save(state, savepath)



if __name__ == '__main__':
    from experiment_config.setup import get_args

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.set_device(0)

    args, modalities = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    import datetime
    now = datetime.datetime.now()
    if args.load == False:

        record_file = open(RECORD_FILE, "w")
        record_file.writelines("Experiment Start: ")
        record_file.writelines(now.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        record_file.close()

    else:
        record_file = open(RECORD_FILE, "a")
        record_file.writelines("Continue Experiment Start: ")
        record_file.writelines(now.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        record_file.close()

    print("logging complete")



   
    model =P4Transformer(radius=0.1, nsamples=32, spatial_stride=32,
                  temporal_kernel_size=3, temporal_stride=2,
                  emb_relu=False,
                  dim=1024, depth=10, heads=8, dim_head=256,
                  mlp_dim=2048, num_classes=17*3, dropout1=0.0, dropout2=0.0)

    # print(sum(param.numel() for param in model.parameters()))
    


    model.to(device=device)

    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, torch.rand(2, 4, 5000, 6).cuda())
    # print(flops.by_module())

    if args.load: #default false
        try:
            # path = os.path.join("checkpoints",  "best_model_fusion.pth")
            path = os.path.join("checkpoints", CHECKPOINT_FILE)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info('Use pretrain model')
        except:
            logging.info('No existing model, starting training from scratch...')



    model.to(device=device)
    print("Model complete")


    try:
        train_dataset = MMPoseLoader('../../../data/mmpose/', split='train', uniform=False, normalized=True,
                                     test_scenario="train", modalities=modalities, 
                                     transforms_type=transforms.Compose([
                     
                      transforms.RandomRotation(15),
                      transforms.Resize([256, 256]),
                      transforms.RandomCrop([224, 224]),
                      transforms.ToTensor()
                  ]))
        test_dataset = MMPoseLoader('../../../data/mmpose/', split='test', uniform=False, normalized=True,
                                    test_scenario=args.test_scene, modalities=modalities 
                                   )
        print("Dataset complete")
    except (AssertionError, RuntimeError, IndexError):
        exit()

    try:
        train_model(
            model=model,
            train_dataset=train_dataset,
            device=device,
            test_dataset=test_dataset,
            modalities=modalities,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            amp=args.amp,
            val_epochs=args.val_epoch,
            l1_flag=args.l1norm,
            l2_flag=args.l2norm,
            scene=args.test_scene
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        # model.use_checkpointing()
        train_model(
            model=model,
            train_dataset= train_dataset,
            device = device,
            test_dataset = test_dataset,
            modalities=modalities,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            amp=args.amp,
            val_epochs=args.val_epoch,
            l1_flag=args.l1norm,
            l2_flag=args.l2norm,
            scene=args.test_scene
        )


