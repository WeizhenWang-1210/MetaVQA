import wandb
from vqa.models.baselines import Baseline
from vqa.models.components import Bert_Encoder, MLP_Multilabel, Resnet50_Encoder, ViT_Encoder, CLIP_ViT_Encoder
from torch.utils.data import DataLoader
from vqa.training.datasets import MultiChoiceDataset
from vqa.qa_preprocessing import answer_space_reversed
from torch import nn
from torch.nn.functional import binary_cross_entropy
import torch
import json
from tqdm import tqdm
import os
from PIL import Image


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


def get_model(MODEL, **kwargs):
    return MODEL(**kwargs)


def get_dataloader(LOADER, split, **kwargs):
    if split == "train":
        return LOADER(
            shuffle=True,
            **kwargs
        )
    else:
        return LOADER(
            shuffle=False,
            **kwargs
        )


def get_dataset(DATASET, **kwargs):
    return DATASET(**kwargs)


def multilabel_eval(network, valid_dl, loss_func, log_qa=False, batch_idx=0, device="cpu"):
    network.to(device)
    """Compute performance of the model on the validation dataset and log a wandb.Table"""
    network.eval()
    val_loss = 0.
    num_correct = 0
    total_prediction = 0
    tp = 0
    p = 0
    with torch.inference_mode():
        with tqdm(valid_dl, unit="batch") as tstep:
            for i, (question, mask, image, lidar, label, index) in enumerate(tstep):
                question, mask, image, lidar, label = question.to(device), mask.to(device), image.to(device), lidar.to(
                    device), label.to(device)
                output = network(question, mask, image, lidar)
                pred = (output > 0.5)
                num_correct += torch.sum(label == pred)
                total_prediction += torch.numel(pred)
                tp += torch.sum((label == 1) & (pred == 1)).item()
                p += torch.sum(label == 1).item()
                val_loss += loss_func(output, label) #* label.size(0)  # So as to get the per-batch loss
                if i == batch_idx and log_qa:
                    qa_ids = []
                    for row, qa_id in enumerate(index):
                        qa_ids.append(qa_id.item())
                    log_qa_table(valid_dl.dataset, qa_ids, pred, label, output)
    return val_loss / len(valid_dl.dataset),num_correct.float()/total_prediction, tp/p


def log_qa_table(dataset, qa_ids, predicted, labels, probs):
    # predicted, labels, probs = predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to


    predicted, labels, probs = predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    table = wandb.Table(columns=["id", "question", "image", "pred", "target", "BCE"])
    for row, id in enumerate(qa_ids):
        qa = dataset.data[id]
        question = qa["question"]
        # print([dataset.map[answer_id] for answer_id, label in enumerate(labels[row]) if label.item() == 1])
        gt = " ".join(
            sorted([str(dataset.map[answer_id]) for answer_id, label in enumerate(labels[row]) if label.item() == 1]))
        preds = " ".join(
            sorted([str(dataset.map[answer_id]) for answer_id, pred in enumerate(predicted[row]) if pred.item() == 1]))
        gt_distro = dataset[id][-2]
        pred_distro = probs[row]
        bce = binary_cross_entropy(gt_distro, pred_distro).item()
        table.add_data(
            id,
            question,
            wandb.Image(os.path.join(dataset.base,qa["rgb"]["front"][0])),
            preds,
            gt,
            bce
        )
    wandb.log({"predictions_table": table})


def multilabel_train(model, train_loader, valid_loader, criterion, optimizer, epochs=20, device="cpu", eval_every=1,
                     save_every=5, log_dir="./", save_best=True, save_last=True):
    os.makedirs(name=f"./{log_dir}", exist_ok=True)
    cur_best = float("-inf")
    model.to(device)
    val_loss = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tstep:
            for question, mask, image, lidar, label, _ in tstep:
                tstep.set_description(f"Epoch {epoch}")
                question, mask, image, lidar, label = question.to(device), mask.to(device), image.to(device), lidar.to(
                    device), label.to(device)
                optimizer.zero_grad()
                output = model(question, mask, image, lidar)
                loss = criterion(output, label)
                loss.backward()
                tstep.set_postfix(loss=loss.item())
                optimizer.step()
                running_loss += loss.item()
        if (epoch + 1) % eval_every == 0:
            avg_loss = running_loss / len(train_loader)
            val_loss, val_accuracy, val_recall = multilabel_eval(model, valid_loader, criterion, log_qa=True, device=device)
            print(
                f'Epoch [{epoch + 1}/{epochs}], Average Train Loss: {avg_loss:.4f}, Average Val Loss, {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Validation Recall: {val_recall:.4f}')
            wandb.log({"acc": val_accuracy,"recall":val_recall, "loss": val_loss})

            if save_best and val_loss < cur_best:
                print('Logging best parameters()')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer': optimizer.__class__.__name__,
                        'val_loss': val_loss,
                    }, os.path.join(log_dir, f'{model.name}_best.pth')
                )
                cur_best = val_loss
        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer': optimizer.__class__.__name__,
                    'val_loss': val_loss,
                }, os.path.join(log_dir, f'{model.name}_{epoch + 1}.pth')
            )
    if save_last:
        torch.save(
            {
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer': optimizer.__class__.__name__,
                'val_loss': val_loss,
            }, os.path.join(log_dir, f'{model.name}_{epochs}.pth')
        )


if __name__ == "__main__":
    base = "/home/weizhen/100k_export"
    split_file = "/home/weizhen/100k_export/split.json"
    qa_file = "/home/weizhen/100k_export/converted.json"
    with open(split_file, "r") as file:
        split_dict = json.load(file)

    wandb_config = dict(
        name="100k_ViT+BERT_2048_1"
    )

    wandb.login()
    wandb.init(
        project="100k_vit",
        name=wandb_config["name"]
    )
    network_config = dict(
        text_encoder=Bert_Encoder(),
        rgb_encoder=ViT_Encoder(),
        predictor=MLP_Multilabel(input_dim=1656, hidden_dim=2048, output_dim=5203, num_hidden=1),
        name="baseline"
    )
    model = get_model(Baseline, **network_config)

    dataloaders = {}
    for split, indices in split_dict.items():
        if split == "src":
            continue
        dataset_config = dict(
            qa_paths=qa_file,
            split=split,
            indices=indices,
            map=answer_space_reversed,
            base = base,
            img_transform = CLIP_ViT_Encoder.get_preprocessor()
        )
        dataset = get_dataset(MultiChoiceDataset, **dataset_config)
        loader_config = dict(
            dataset=dataset,
            batch_size=128
        )
        loader = get_dataloader(DataLoader, dataset.split, **loader_config)
        dataloaders[split] = loader
        print(f"{split}:{len(dataset)}")



    model = get_model(Baseline, **network_config)
    get_model_size(model)
    bceloss = torch.nn.BCELoss(reduction = 'sum')
    adam = torch.optim.Adam(model.parameters(), lr=1e-5)
    multilabel_train(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["val"],
        criterion=bceloss,
        optimizer=adam,
        epochs=50,
        log_dir=wandb_config["name"],
        device="cuda",

    )
    # multilabel_eval(model,dataloaders["test"],nn.)
    #multilabel_eval(model, dataloaders["test"], nn.BCELoss(), log_qa=True, device="cuda:7")
    wandb.finish()
