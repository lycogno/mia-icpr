import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from attackers import MiaAttack
import os
import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils1 import train_model, save_model, load_model, evaluate_model, create_classification_report
from data import CIFAR10Data
from module import CIFAR10Module
from train import *
import torchvision
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--prune_epochs', default=50, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str, help="prune method for victim model")
parser.add_argument('--prune_sparsity', default=0.5, type=float, help="prune sparsity for victim model")
parser.add_argument('--adaptive', action='store_true', help="use adaptive attack")
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia", type=str)
parser.add_argument('--original', action='store_true', help="original=true, then launch attack against original model")

# PROGRAM level args
parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
parser.add_argument(
    "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
)
parser.add_argument("--pruned_weights", type=str, default="resnet18_pruned_model_1.pt")

# TRAINER args
parser.add_argument("--classifier", type=str, default="resnet18")
parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpu_id", type=str, default="3")

parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=1e-2)


num_classes = 10
random_seed = 1
l1_regularization_strength = 1e-4
l2_regularization_strength = 1e-4
learning_rate = 1e-3
learning_rate_decay = 1

cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

model_dir = "saved_models_afterprune"
model_filename = "resnet18_cifar10.pt"
model_filename_prefix = "resnet18_pruned_model"
pruned_model_filename = "resnet18_afterprune.pt"
model_filepath = os.path.join(model_dir, model_filename)
pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
transform = T.Compose(
[
    T.ToTensor(),
    T.Normalize(mean, std),
]
)
test_data=torchvision.datasets.ImageFolder(root='/home/sameenahmad/aryan1/fine_tune_dataset/sorted',transform=transform)

def test(self, test_loader, log_pref=""):
    self.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCELoss() if isinstance(self, nn.Sequential) else nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            if isinstance(criterion, nn.BCELoss):
                correct += torch.sum(torch.round(outputs) == targets)
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total
    total_loss /= total
    if log_pref:
        print("Correct: {}, Total: {}".format(correct, total))
        print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
    return acc, total_loss

def main(args):
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    # prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}" \
    #                f"{'_' + args.defend if args.defend else ''}{'_' + str(args.defend_arg) if args.defend else ''}"
    # prune_prefix2 = f"{args.pruner_name}_{args.prune_sparsity}" \
    #                 f"{'_' + args.defend if args.adaptive else ''}{'_' + str(args.defend_arg) if args.adaptive else ''}"

    save_folder = f"../mia_prune/results/{args.dataset_name}_{args.model_name}"

    print(f"Save Folder: {save_folder}")

    # Load datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    # if testset is None:
    #     total_dataset = trainset
    # else:
    #     total_dataset = ConcatDataset([trainset, testset])
    total_dataset = testset
    total_size = len(total_dataset)
    victim_train_dataset = Subset(trainset, np.random.choice(len(trainset), 10000, replace=False))
    victim_test_dataset = testset
    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(trainset)}, "
          f"Victim Test Size: {len(testset)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # Load origina; victim model
    victim_model_save_folder = save_folder + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    # WARNING: NOT LOADING ANY WEIGHTS
    # victim_model.load(victim_model_path)
    sparsity = [.5, .6, .75]
    epoch_count = [20, 30, 40]
    for prune_sparsity in [.75]:
        for num_epochs in epoch_count:
            for shadow_prune_sparsity in sparsity:
                if shadow_prune_sparsity == prune_sparsity: continue
                pruned_model_filename = f"{args.classifier}_pruned_{prune_sparsity}_{num_epochs}.pt"
                pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)
                # Load pruned and finetuned victim models
                model = CIFAR10Module(args)
                # data = CIFAR10Data(args)
                state_dict = pruned_model_filepath
                # FOR RESNET ONLY
                # for module_name, module in  model.model.named_modules():
                #     if 'conv' in module_name or 'layer3.0.downsample.0' in module_name or 'layer2.0.downsample.0' in module_name or 'layer4.0.downsample.0' in module_name:
                #         try:
                #             prune.identity(module, name="weight")
                #         except:
                #             pass
                print(f"Load Pruned Victim Model From {state_dict}")
                model.load_state_dict(torch.load(state_dict))
                victim_pruned_model=model.model
                victim_pruned_model.to(cuda_device)
                # print(len(victim_train_loader), victim_train_loader.__iter__().next()[0].shape, victim_train_loader.__iter__().next()[1].shape)
                test(victim_pruned_model, victim_train_loader, "Train Victim Model")
                test(victim_pruned_model, victim_test_loader, "Test Victim Model")

                # Load pruned shadow models
                shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list = [], [], [], []
                for shadow_ind in range(args.shadow_num):
                    shadow_train_dataset = Subset(trainset, np.random.choice(len(trainset), 10000, replace=False))
                    shadow_dev_dataset = testset
                    shadow_test_dataset = testset
                    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                    pin_memory=False)
                    shadow_dev_loader = DataLoader(shadow_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                pin_memory=False)
                    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                    pin_memory=False)

                    shadow_model_path = f"{save_folder}/shadow_model_{shadow_ind}/best.pth"
                    shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
                    # shadow_model.load(shadow_model_path)
                    model = CIFAR10Module(args)
                    # for module_name, module in  model.model.named_modules():
                    #     try:
                    #         prune.identity(module, name="weight")
                    #     except:
                    #         pass
                    #     try:
                    #         prune.identity(module, name="weight")
                    #     except:
                    #         pass
                    state_dict = f"shadow_models_afterprune/shadow_{shadow_ind}{args.classifier}_pruned_{shadow_prune_sparsity}_{num_epochs}.pt"
                    print(f"Load Pruned Shadow Model From {state_dict}")
                    model.load_state_dict(torch.load(state_dict))
                    shadow_pruned_model=model.model
                    shadow_pruned_model.to(cuda_device)
                    test(shadow_pruned_model, shadow_train_loader, "Shadow Pruned Model Train")
                    test(shadow_pruned_model, shadow_test_loader, "Shadow Pruned Model Test")
                    shadow_model_list.append(shadow_model)
                    shadow_prune_model_list.append(shadow_pruned_model)
                    shadow_train_loader_list.append(shadow_train_loader)
                    shadow_test_loader_list.append(shadow_test_loader)

                print("Start Membership Inference Attacks")

                if args.original:
                    attack_original = True
                else:
                    attack_original = False
                attacker = MiaAttack(
                    victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
                    shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list,
                    num_cls=args.num_cls, device=device, batch_size=args.batch_size,
                    attack_original=attack_original)

                attacks = args.attacks.split(',')

                if "samia" in attacks:
                    nn_trans_acc = attacker.nn_attack("nn_sens_cls", model_name="transformer")
                    print(f"SAMIA attack accuracy {nn_trans_acc:.3f}")

                if "threshold" in attacks:
                    conf, xent, mentr, top1_conf = attacker.threshold_attack()
                    print(f"Ground-truth class confidence-based threshold attack (Conf) accuracy: {conf:.3f}")
                    print(f"Cross-entropy-based threshold attack (Xent) accuracy: {xent:.3f}")
                    print(f"Modified-entropy-based threshold attack (Mentr) accuracy: {mentr:.3f}")
                    print(f"Top1 Confidence-based threshold attack (Top1-conf) accuracy: {top1_conf:.3f}")

                if "nn" in attacks:
                    nn_acc = attacker.nn_attack("nn")
                    print(f"NN attack accuracy {nn_acc:.3f}")

                if "nn_top3" in attacks:
                    nn_top3_acc = attacker.nn_attack("nn_top3")
                    print(f"Top3-NN Attack Accuracy {nn_top3_acc}")

                if "nn_cls" in attacks:
                    nn_cls_acc = attacker.nn_attack("nn_cls")
                    print(f"NNCls Attack Accuracy {nn_cls_acc}")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print(args)
    import time
    start_time = time.time()
    main(args)
    print("--- Loaded models and ran attacks in %.2f seconds ---" % (time.time() - start_time))

    # Create an untrained model.
    # model = create_model(num_classes=num_classes)

    # Load a pretrained model.
    # model = load_model(model=model,
                    #    model_filepath=model_filepath,
                    #    device=cuda_device)


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def iterative_pruning_finetuning(model,
                                 train_loader,
                                 test_loader,
                                 device,
                                 learning_rate,
                                 l1_regularization_strength,
                                 l2_regularization_strength,
                                 learning_rate_decay=0.1,
                                 conv2d_prune_amount=0.2,
                                 linear_prune_amount=0,
                                 num_iterations=10,
                                 num_epochs_per_iteration=10,
                                 model_filename_prefix="pruned_model",
                                 model_dir="saved_models",
                                 grouped_pruning=False):

    for i in range(num_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=linear_prune_amount)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        # print(model.conv1._forward_pre_hooks)

        print("Fine-tuning...")

        train_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    l1_regularization_strength=l1_regularization_strength,
                    l2_regularization_strength=l2_regularization_strength,
                    learning_rate=learning_rate * (learning_rate_decay**i),
                    num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = evaluate_model(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
        model_filepath = os.path.join(model_dir, model_filename)
        save_model(model=model,
                   model_dir=model_dir,
                   model_filename=model_filename)
        model = load_model(model=model,
                           model_filepath=model_filepath,
                           device=device)

    return model

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

