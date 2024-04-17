import os
import copy
import torch
import torch.nn.utils.prune as prune
from utils1 import set_random_seeds, create_model, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report
from data import CIFAR10Data
from module import CIFAR10Module
from train import *
import torchvision
import torchvision.transforms as T
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
    model = remove_parameters(model=model)
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


def main():
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    
    num_classes = 10
    random_seed = 1
    l1_regularization_strength = 1e-4
    l2_regularization_strength = 1e-4
    learning_rate = 1e-3
    learning_rate_decay = 1

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models_afterprune"
    model_filename = args.classifier+".pt"
    model_filename_prefix = args.classifier
    model_filepath = os.path.join(model_dir, model_filename)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    )
    test_data=torchvision.datasets.ImageFolder(root='/home/sameenahmad/aryan1/fine_tune_dataset/sorted',transform=transform)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    # model = create_model(num_classes=num_classes)

    # Load a pretrained model.
    # model = load_model(model=model,
                    #    model_filepath=model_filepath,
                    #    device=cuda_device)
    sparsity = [.5, .6, .75]
    epoch_count = [20, 30, 40]
    for prune_sparsity in sparsity:
        for num_epochs in epoch_count:
            pruned_model_filename = f"{args.classifier}_pruned_{prune_sparsity}_{num_epochs}.pt"
            pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)
            print(f"Generating model: {pruned_model_filepath}")
            print(f"Pruning sparsity: {prune_sparsity} Number of epochs: {num_epochs}")
            model = CIFAR10Module(args)
            # data = CIFAR10Data(args)
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))
            resmodel=model.model
            train_loader, test_loader, classes = prepare_dataloader(
                num_workers=8, train_batch_size=256, eval_batch_size=256)

            _, eval_accuracy = evaluate_model(model=resmodel,
                                            test_loader=test_loader,
                                            device=cuda_device,
                                            criterion=None)
            print(eval_accuracy)
            classification_report = create_classification_report(
                model=resmodel, test_loader=test_loader, device=cuda_device)

            num_zeros, num_elements, sparsity = measure_global_sparsity(model)

            print("Test Accuracy: {:.3f}".format(eval_accuracy))
            print("Classification Report:")
            print(classification_report)
            print("Global Sparsity:")
            print("{:.2f}".format(sparsity))

            print("Iterative Pruning + Fine-Tuning...")

            pruned_model = copy.deepcopy(model)

            pruned_model = iterative_pruning_finetuning(
                model=pruned_model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=cuda_device,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay,
                l1_regularization_strength=l1_regularization_strength,
                l2_regularization_strength=l2_regularization_strength,
                conv2d_prune_amount=prune_sparsity,
                linear_prune_amount=0,
                num_iterations=1,
                num_epochs_per_iteration=num_epochs,
                model_filename_prefix=model_filename_prefix,
                model_dir=model_dir,
                grouped_pruning=True)

            # Apply mask to the parameters and remove the mask.
            pruned_model = remove_parameters(model=pruned_model)

            _, eval_accuracy = evaluate_model(model=pruned_model,
                                            test_loader=test_loader,
                                            device=cuda_device,
                                            criterion=None)

            classification_report = create_classification_report(
                model=pruned_model, test_loader=test_loader, device=cuda_device)

            num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

            print("Test Accuracy: {:.3f}".format(eval_accuracy))
            print("Classification Report:")
            print(classification_report)
            print("Global Sparsity:")
            print("{:.2f}".format(sparsity))

            save_model(model=pruned_model, model_dir=model_dir, model_filename=pruned_model_filename)


if __name__ == "__main__":

    main()