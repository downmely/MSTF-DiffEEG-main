from trainer import Diffusion
from dataset.data_pre import load_MODMA_data, load_PREDCT_data, load_DEAP_data, load_SEED_data, load_REFED_data, augment_data
import time
import torch
import numpy as np
import argparse
import random
import time
from torch import cuda
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold



def clear_gpu_memory():
    if cuda.is_available():
        torch.cuda.empty_cache() 
        cuda.ipc_collect()      

def GetNowTime():
    return time.strftime("%m%d%H%M%S",time.localtime(time.time()))
            
def set_all(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.device:
        torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(0)
    
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2025, type=int)
parser.add_argument('--device', type=bool, default=torch.cuda.is_available())  
parser.add_argument('--cpu',  action='store_true', help='Ignore CUDA.')

# data
parser.add_argument('--data_dataset', type=str, default='REFED', help='Path to the dataset.')
parser.add_argument('--data_path1', type=str, default='./dataset/MODMA/all_HC_DE.mat')
parser.add_argument('--data_path2', type=str, default='./dataset/MODMA/all_MDD_DE.mat')
parser.add_argument('--data_path3', type=str, default='./dataset/PREDCT/all_DE.mat')
parser.add_argument('--data_path4', type=str, default='./dataset/DEAP')
parser.add_argument('--deap_label', type=str, default='A', choices=['A', 'V', 'AV'])
parser.add_argument('--data_path5', type=str, default='./dataset/SEED/DE_data.mat')
parser.add_argument('--data_path6', type=str, default='./dataset/REFED/DE_allbands.mat')
parser.add_argument('--refed_label', type=str, default='A', choices=['A', 'V', 'AV'])
parser.add_argument('--data_augment', action="store_false")
parser.add_argument('--data_augment_method', type=str, default="gaussian_noise",choices=['gaussian_noise', 'time_masking', 'phase_shuffling', 'random_crop'])
parser.add_argument('--data_augment_times', type=int, default=2, help='Times of augmented data to original data.')

# model
parser.add_argument('--model_var_type', type=str, default='fixedlarge')
parser.add_argument('--model_embed_dim', type=int, default=16)
parser.add_argument('--model_depth', type=int, default=3)
parser.add_argument('--model_heads', type=int, default=4)
parser.add_argument('--model_ema_rate', type=float, default=0.9999)
parser.add_argument('--model_ema', type=bool, default=False)
parser.add_argument('--model_pred_MS', type=int, default=1)
parser.add_argument('--model_UNet_MS', type=int, default=1)
parser.add_argument('--model_pred_CA', type=int, default=1)
parser.add_argument('--model_UNet_CA', type=int, default=1)

# diffusion
parser.add_argument('--diffusion_S_noise_prior', type=int, default=1)
parser.add_argument('--diffusion_T_noise_prior', type=int, default=1)
parser.add_argument('--diffusion_F_noise_prior', type=int, default=1)
parser.add_argument('--diffusion_timesteps', type=int, default=100)
parser.add_argument('--diffusion_beta_schedule', type=str, default='linear')
parser.add_argument('--diffusion_beta_start', type=float, default=0.0001)
parser.add_argument('--diffusion_beta_end', type=float, default=0.02)
parser.add_argument('--diffusion_pred_model_joint_train', action="store_false")

# training
parser.add_argument('--training_batch_size', type=int, default=32)
parser.add_argument('--trained_pred_model_path', type=str, default='saved_models')
parser.add_argument('--train_pred_model', type=bool, default=True)
parser.add_argument('--train_pred_model_epochs', type=int, default=100)
parser.add_argument('--train_pred_model_logging_interval', type=int, default=1)
parser.add_argument("--train_guidance_only", action="store_true", help="Whether to only pre-train the guidance classifier f_phi")
parser.add_argument("--resume_training", action="store_true", help="Whether to resume training")
parser.add_argument("--resume_pred_training", action="store_true", help="Whether to resume training")
parser.add_argument('--training_num_epochs', type=int, default=2000)
parser.add_argument('--training_lambda_ce', type=float, default=0.5)
parser.add_argument('--training_logging_freq', type=int, default=100)
parser.add_argument('--training_snapshot_freq', type=int, default=10000000000)
parser.add_argument('--training_validation_freq', type=int, default=100)
parser.add_argument('--training_validation_freq_pred', type=int, default=10)
parser.add_argument('--training_warmup_epochs', type=int, default=40)

# testing
parser.add_argument('--testing_batch_size', type=int, default=32)
parser.add_argument("--eval_best",     action="store_false", help="Evaluate best model during training, instead of the ckpt stored at the last epoch")
parser.add_argument('--test_tsne', type=bool, default=False)

# optim
parser.add_argument('--optim', choices=['SGD', 'RMSProp', 'Adam'], default='Adam', help='Optimizer: SGD, RMSProp or Adam.')
parser.add_argument('--optim_lr_schedule', type=bool, default=True)
parser.add_argument('--optim_grad_clip', type=float, default=1.0)
parser.add_argument('--optim_weight_decay', type=float, default=0.000)
parser.add_argument('--optim_optimizer', type=str, default="Adam")
parser.add_argument('--optim_lr', type=float, default=0.001)
parser.add_argument('--optim_beta1', type=float, default=0.9)
parser.add_argument('--optim_amsgrad', type=bool, default=False)
parser.add_argument('--optim_eps', type=float, default=0.00000001)
parser.add_argument('--optim_min_lr', type=float, default=0.0)

args = parser.parse_args()

set_all(args)

init_time = time.time()
    
#####Dataloader 
if args.data_dataset == 'MODMA':
    data, labels = load_MODMA_data(args.data_path1, args.data_path2)
elif args.data_dataset == 'PREDCT':
    data, labels = load_PREDCT_data(args.data_path3) 
elif args.data_dataset == 'DEAP':
    data, labels = load_DEAP_data(args.data_path4, args.deap_label)
elif args.data_dataset == 'SEED':
    data, labels = load_SEED_data(args.data_path5)  
elif args.data_dataset == 'REFED':
    data, labels = load_REFED_data(args.data_path6, args.refed_label)

if args.data_augment:
    data, labels = augment_data(data, labels, args.data_augment_method, args.data_augment_times)
print(f"data_shape:{data.shape}")
print(f"labels_shape:{labels.shape}")
print(f"label type:{args.deap_label}   {args.refed_label}")

global_start_time = time.time()

K = 5  
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

all_results = []
fold_times = []

result_file_path = "result.txt"
with open(result_file_path, "a") as f:
    f.write("\n{:=^60}\n".format(" CV Results "))
    f.write("# Hyperparams:\n")
    f.write(f"# dataset={args.data_dataset}, aug={args.data_augment}, depth={args.model_depth}, steps={args.diffusion_timesteps}\n")
    f.write(f"# joint_train={args.diffusion_pred_model_joint_train}, pred_epochs={args.train_pred_model_epochs}, epochs={args.training_num_epochs}, eval_best={args.eval_best}\n")
    f.write(f"# aug_times={args.data_augment_times}, aug_method={args.data_augment_method}\n")
    # 简约的表头
    f.write("Fold\tPre_Acc\tJoint_Acc\tAcc\tSen\tSpe\tF1\tPrec\tRec\tTime(min)\n")
    f.write("-"*80 + "\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):  
    clear_gpu_memory()  
    fold_start_time = time.time()
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    print(f'Fold {fold}:')
    print('Train:', train_data.size(), 'Test:', test_data.size())

    ## 准备好模型
    model = Diffusion(args)
    pretrain_train_acc, pretrain_test_acc, joint_train_train_acc, joint_train_test_acc = model.train(
        fold, train_data, train_labels, test_data, test_labels)

    result = model.test(fold, test_data, test_labels)
    
    fold_end_time = time.time()
    fold_time = (fold_end_time - fold_start_time)/60
    fold_times.append(fold_time)
    
    fold_results = {
        'fold': fold,
        'pretrain_acc': pretrain_test_acc,
        'joint_train_acc': joint_train_test_acc,
        'test_acc': result[0],
        'sensitivity': result[1],
        'specificity': result[2],
        'f1': result[3],
        'precision': result[4],
        'recall': result[5],
        'time': fold_time
    }
    all_results.append(fold_results)
    
    print("{:<6} {:<8.4f} {:<8.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<8.2f}".format(
        fold, pretrain_test_acc, joint_train_test_acc, result[0], 
        result[1], result[2], result[3], result[4], result[5], fold_time))

    with open(result_file_path, "a") as f:
        f.write(f"{fold}\t{pretrain_test_acc:.4f}\t{joint_train_test_acc:.4f}\t{result[0]:.4f}\t{result[1]:.4f}\t{result[2]:.4f}\t{result[3]:.4f}\t{result[4]:.4f}\t{result[5]:.4f}\t{fold_time:.2f}\n")

mean_pretrain = np.mean([r['pretrain_acc'] for r in all_results])
mean_joint = np.mean([r['joint_train_acc'] for r in all_results])
mean_test_acc = np.mean([r['test_acc'] for r in all_results])
mean_sensitivity = np.mean([r['sensitivity'] for r in all_results])
mean_specificity = np.mean([r['specificity'] for r in all_results])
mean_f1 = np.mean([r['f1'] for r in all_results])
mean_precision = np.mean([r['precision'] for r in all_results])
mean_recall = np.mean([r['recall'] for r in all_results])
mean_time = np.mean(fold_times)

print("-"*80)
print("{:<6} {:<8.4f} {:<8.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<6.4f} {:<8.2f}".format(
    "Mean", mean_pretrain, mean_joint, mean_test_acc, 
    mean_sensitivity, mean_specificity, mean_f1, mean_precision, mean_recall, mean_time))
print("="*80)

with open(result_file_path, "a") as f:
    f.write(f"\n# Mean\n")
    f.write("Mean\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\n".format(
        mean_pretrain, mean_joint, mean_test_acc, 
        mean_sensitivity, mean_specificity, mean_f1, mean_precision, mean_recall, mean_time))
    f.write("="*80 + "\n")

global_end_time = time.time()
print(f"\nTotal time: {(global_end_time - global_start_time)/60:.2f} min")
print(f"Average fold time: {mean_time:.2f} min")