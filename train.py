import os 
import time
from utils.generate import select_device
import wandb
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.config import *
from model.anchor import Anchors
from utils.mlops_tool import use_data_wandb
from model.model import RetinaFace, forward
from utils.data_tool import create_exp_dir
from model.multibox_loss import MultiBoxLoss
from model.metric import calculate_map, calculate_running_map
from utils.dataset import WiderFaceDataset, LatentWiderFaceDataset, detection_collate

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument('--run', type=str, default=RUN_NAME, help="run name")
    parser.add_argument('--epoch', type=int, default=EPOCHS, help="number of epoch")
    parser.add_argument('--image_size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--model', type=str, default='resnet50', help='select model')
    parser.add_argument('--freeze', action='store_true', help="freeze model backbone")
    parser.add_argument('--weight', type=str, default=None, help='path to pretrained weight')
    parser.add_argument('--weight_decay', type=int, default=WEIGHT_DECAY, help="weight decay of optimizer")
    parser.add_argument('--momentum', type=int, default=MOMENTUM, help="momemtum of optimizer")
    parser.add_argument('--startfm', type=int, default=START_FRAME, help="architecture start frame")
    parser.add_argument('--batchsize', type=int, default=BATCH_SIZE, help="total batch size for all GPUs (default:")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="init learning rate (default: 0.0001)")
    parser.add_argument('--download', action='store_true', help="download dataset from Wandb Database")
    parser.add_argument('--tuning', action='store_true', help="no plot image for tuning")
    parser.add_argument('--device', type=str, default='', help="no plot image for tuning")
    parser.add_argument('--use_latent', action='store_true', help="use latent representation as input instead of RGB images")
    parser.add_argument('--latent_dir', type=str, default='latent', help="directory containing latent representation files")
    parser.add_argument('--label_file', type=str, default='labels.txt', help="file containing annotations")
    parser.add_argument('--latent_suffix', type=str, default='latent_75.npy', help="specific latent file to use")
    parser.add_argument('--debug', action='store_true', help="enable debug mode with shapes printed")
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help="path to dataset directory")

    args = parser.parse_args()
    return args

def train(model, anchors, trainloader, optimizer, loss_function, device='cpu'):
    model.train()
    loss_cls, loss_box, loss_pts = 0, 0, 0
    
    for i, (input, targets) in enumerate(trainloader):
        # load data into cuda
        input   = input.to(device)
        targets = [annos.to(device) for annos in targets]

        # forward + backpropagation + step
        loss_l, loss_c, loss_landm = forward(model, input, targets, anchors, loss_function, optimizer)

        # metric
        loss_cls += loss_c
        loss_box += loss_l 
        loss_pts += loss_landm

        # free after backward
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # cls = classification; box = box regressionl; pts = landmark regression
    loss_cls = loss_cls/len(trainloader)
    loss_box = loss_box/len(trainloader)
    loss_pts = loss_pts/len(trainloader)

    return loss_cls, loss_box, loss_pts

def evaluate(model, anchors, validloader, loss_function, best_box, device='cpu'):
    model.eval()
    loss_cls, loss_box, loss_pts = 0, 0, 0
    count_img, count_target = 0, 0
    ap_5, ap_5_95 = 0, 0

    with torch.no_grad():
        for i, (input, targets) in enumerate(validloader):
            # load data into cuda
            input   = input.to(device)
            targets = [annos.to(device) for annos in targets]

            # forward
            predict = model(input)
            loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)

            # metric
            loss_cls += loss_c
            loss_box += loss_l 
            loss_pts += loss_landm

            # bap_5, bap_5_95 = calculate_running_map(targets, predict)
            # ap_5    += bap_5
            # ap_5_95 += bap_5_95

            # summary
            count_img += input.shape[0]
            for target in targets:
                count_target += target.shape[0]
    
    loss_cls = loss_cls/len(validloader)
    loss_box = loss_box/len(validloader)
    loss_pts = loss_pts/len(validloader)

    epoch_ap_5 = ap_5/len(validloader)
    epoch_ap_5_95 = ap_5_95/len(validloader)

    epoch_summary = [count_img, count_target, epoch_ap_5, epoch_ap_5_95]

    if loss_box>best_box:
    # export to onnx + pt
        torch.save(model.state_dict(), os.path.join(save_dir, 'weight.pth'))

    return loss_cls, loss_box, loss_pts, epoch_summary

if __name__ == '__main__':
    args = parse_args()

    # Kiểm tra thư mục hiện tại và cấu trúc thư mục
    print("\nĐang kiểm tra cấu trúc thư mục:")
    import os
    
    # Lưu thư mục hiện tại
    current_dir = os.getcwd()
    print(f"Thư mục hiện tại: {current_dir}")
    
    # Tìm thư mục data
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Không tìm thấy thư mục dữ liệu tại {data_path}!")
        
        # Kiểm tra các đường dẫn tuyệt đối thường dùng trong Kaggle
        kaggle_paths = [
            "/kaggle/working/Retina_modify1/data",
            "/kaggle/working/data",
            "/kaggle/input/data"
        ]
        
        for path in kaggle_paths:
            if os.path.exists(path):
                data_path = path
                args.data_path = path
                print(f"Đã tìm thấy thư mục dữ liệu tại: {path}")
                break
        
        # Tìm ở thư mục cha
        if not os.path.exists(data_path):
            parent_dir = os.path.dirname(current_dir)
            potential_data_path = os.path.join(parent_dir, "data")
            if os.path.exists(potential_data_path):
                data_path = potential_data_path
                args.data_path = potential_data_path
                print(f"Đã tìm thấy thư mục dữ liệu tại: {potential_data_path}")
            
            # Tìm ở thư mục cha của cha
            if not os.path.exists(data_path):
                grandparent_dir = os.path.dirname(parent_dir)
                potential_data_path = os.path.join(grandparent_dir, "data")
                if os.path.exists(potential_data_path):
                    data_path = potential_data_path
                    args.data_path = potential_data_path
                    print(f"Đã tìm thấy thư mục dữ liệu tại: {potential_data_path}")
    
    # Nếu đã tìm thấy thư mục data, kiểm tra cấu trúc
    if os.path.exists(data_path):
        print(f"\nNội dung trong thư mục dữ liệu {data_path}:")
        try:
            for f in os.listdir(data_path):
                f_path = os.path.join(data_path, f)
                if os.path.isdir(f_path):
                    print(f"  [Thư mục] {f}")
                    # Liệt kê nội dung thư mục con
                    try:
                        subfiles = os.listdir(f_path)
                        for sf in subfiles[:5]:  # Chỉ hiển thị 5 file/thư mục đầu tiên
                            sf_path = os.path.join(f_path, sf)
                            if os.path.isdir(sf_path):
                                print(f"    [Thư mục] {sf}")
                                # Liệt kê một số file trong thư mục con
                                try:
                                    subsubfiles = os.listdir(sf_path)
                                    if subsubfiles:
                                        print(f"      Ví dụ: {', '.join(subsubfiles[:3])}" + ("..." if len(subsubfiles) > 3 else ""))
                                except:
                                    pass
                            else:
                                print(f"    [File] {sf}")
                        if len(subfiles) > 5:
                            print(f"    ... và {len(subfiles) - 5} file/thư mục khác")
                    except Exception as e:
                        print(f"    Lỗi: {e}")
                else:
                    print(f"  [File] {f}")
        except Exception as e:
            print(f"Lỗi khi liệt kê nội dung thư mục: {e}")
    else:
        print(f"\nKhông tìm thấy thư mục dữ liệu ở bất kỳ vị trí nào!")
        print("Vui lòng chỉ định đường dẫn tuyệt đối đến thư mục dữ liệu với --data_path")
    
    # Cập nhật các đường dẫn cho train và validation
    args.train_latent_path = os.path.join(args.data_path, TRAIN_PATH, args.latent_dir)
    args.val_latent_path = os.path.join(args.data_path, VALID_PATH, args.latent_dir)
    args.train_label_path = os.path.join(args.data_path, TRAIN_PATH, args.label_file)
    args.val_label_path = os.path.join(args.data_path, VALID_PATH, args.label_file)
    
    print(f"\nĐường dẫn được sử dụng:")
    print(f"  Train latent: {args.train_latent_path}")
    print(f"  Train labels: {args.train_label_path}")
    print(f"  Val latent: {args.val_latent_path}")
    print(f"  Val labels: {args.val_label_path}")
    
    # init wandb
    config = dict(
        epoch           = args.epoch,
        weight_decay    = args.weight_decay,
        momentum        = args.momentum,
        lr              = args.lr,
        batchsize       = args.batchsize,
        startfm         = args.startfm,
        input_size      = args.image_size,
        use_latent      = args.use_latent,
        latent_suffix   = args.latent_suffix
    )
    
    # log experiments to
    run = wandb.init(project=PROJECT, config=config)
    
    # use artifact - make it optional
    try:
        if args.download:
            use_data_wandb(run=run, data_name=DATASET, download=args.download)
    except Exception as e:
        print(f"Warning: Could not download wandb artifact: {e}")
        print("Continuing with local data...")

    # train on device
    device = select_device(args.device, args.batchsize)

    # get dataloader
    if args.use_latent or USE_LATENT:
        # Use the latent representation dataset
        print(f"\tUsing latent representations with suffix: {args.latent_suffix}")
        train_set = LatentWiderFaceDataset(
            root_path=args.data_path, 
            latent_dir=args.latent_dir, 
            label_file=args.label_file,
            latent_suffix=args.latent_suffix,
            is_train=True
        )
        valid_set = LatentWiderFaceDataset(
            root_path=args.data_path, 
            latent_dir=args.latent_dir, 
            label_file=args.label_file,
            latent_suffix=args.latent_suffix,
            is_train=False
        )
    else:
        # Use the regular RGB image dataset
        train_set = WiderFaceDataset(root_path=DATA_PATH, input_size=args.image_size, is_train=True)
        valid_set = WiderFaceDataset(root_path=DATA_PATH, input_size=args.image_size, is_train=False)
    
    print(f"\tNumber of training example: {len(train_set)}\n\tNumber of validation example: {len(valid_set)}")

    torch.manual_seed(RANDOM_SEED)

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detection_collate)
    validloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detection_collate)

    n_classes = N_CLASSES
    epochs = args.epoch
    # create dir for save weight
    save_dir = create_exp_dir()

    # get model and define loss func, optimizer
    model = RetinaFace(
        model_name=args.model, 
        freeze_backbone=args.freeze,
        use_latent=args.use_latent or USE_LATENT,
        debug=args.debug
    ).to(device)
    
    if args.weight is not None and os.path.isfile(args.weight):
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint)
        print(f'\tWeight located in {args.weight} have been loaded')

    cudnn.benchmark = True

    print("\nGenerating anchors...")
    print(f"Model feature_map configuration: {model.feature_map}")
    print(f"Using latent mode: {args.use_latent or USE_LATENT}")
    
    with torch.no_grad():
        anchors = Anchors(pyramid_levels=model.feature_map).forward().to(device)
        
    print(f"Generated {anchors.size(0)} anchors\n")

    # optimizer + citeration
    optimizer   = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion   = MultiBoxLoss(N_CLASSES, 
                    overlap_thresh=OVERLAP_THRES, 
                    prior_for_matching=True, 
                    bkg_label=BKG_LABEL, neg_pos=True, 
                    neg_mining=NEG_MINING, neg_overlap=NEG_OVERLAP, 
                    encode_target=False, device=device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONE, gamma=0.7)

    # wandb watch
    run.watch(models=model, criterion=criterion, log='all', log_freq=10)

    # training
    best_ap = -1

    for epoch in range(epochs):
        print(f'\n\tEpoch\tbox\t\tlandmarks\tcls\t\ttotal')
        t0 = time.time()
        loss_cls, loss_box, loss_pts = train(model, anchors, trainloader, optimizer, criterion, device)
        t1 = time.time()

        total_loss = loss_box + loss_pts + loss_cls
        # epoch
        wandb.log({'loss_cls': loss_cls, 'loss_box': loss_box, 'loss_landmark': loss_pts}, step=epoch)
        print(f'\t{epoch+1}/{epochs}\t{loss_box:.5f}\t\t{loss_pts:.5f}\t\t{loss_cls:.5f}\t\t{total_loss:.5f}\t\t{(t1-t0):.2f}s')
        
        # summary [count_img, count_target, epoch_ap_5, epoch_ap_5_95]
        t0 = time.time()
        loss_cls, loss_box, loss_pts, summary = evaluate(model, anchors, validloader, criterion, loss_box, device)
        t1 = time.time()

        # images, labels, P, R, map_5, map_95
        print(f'\n\tImages\tLabels\t\tbox\t\tlandmarks\tcls\t\tmAP@.5\t\tmAP.5.95')
        # print(f'\t{summary[0]}\t{summary[1]}\t\t{summary[2]}\t\t{summary[3]}')
        print(f'\t{summary[0]}\t{summary[1]}\t\t{loss_box:.5f}\t\t{loss_pts:.3f}\t\t{loss_cls:.5f}\t\t{(t1-t0):.2f}s')
    
        wandb.log({'val.loss_cls': loss_cls, 'val.loss_box': loss_box, 'val.loss_landmark': loss_pts}, step=epoch)
        wandb.log({'metric.map@.5': summary[2], 'metric.map@.5:.95': summary[3]}, step=epoch)
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
        
        # decrease lr
        scheduler.step()

        # Wandb summary
        # if summary[2] > best_ap:
        #     best_ap = summary[2] 
        #     wandb.run.summary["best_accuracy"] = best_ap

    if not args.tuning:
        trained_weight = wandb.Artifact(args.run, type='WEIGHTS')
        # trained_weight.add_file(os.path.join(save_dir, 'weight.onnx'))
        trained_weight.add_file(os.path.join(save_dir, 'weight.pth'))
        wandb.log_artifact(trained_weight)