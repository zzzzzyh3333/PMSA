"""Preserved sequential Trainer, extracted from the validated weighted runner.

The separate validation monitor pass and randomly initialized BEST model are
retained because they consume random numbers in the historical training loop.
Only train/val data are supplied here. Test evaluation is a separate command.
"""
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from Model import make_model as ScaleBankRC
from Model.Model import ClassSlotRelationConsolidation
from utils.Continual import average_accuracy, forgetting, performance_aware_forgetting
from utils.recorder import Recorder


def data_generator(config, task):
    trainset = Load_Dataset(_data['train'][task])
    valset = Load_Dataset(_data['val'][task])
    train = DataLoader(trainset, batch_size=config.batchsize, shuffle=True, drop_last=False, num_workers=0)
    val = DataLoader(valset, batch_size=config.eval_batchsize, shuffle=False, drop_last=False, num_workers=0)
    monitor = DataLoader(valset, batch_size=config.eval_batchsize, shuffle=False, drop_last=False, num_workers=0)
    return train, val, monitor


def train(config, data, output):
    # The original Trainer helpers use module-level args; runs are sequential.
    global args, _data, _record
    args, _data = config, data
    _record = Recorder(output)
    model = ScaleBankRC(args).float().to(args.device)
    return Trainer(args, model)


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        if args.model == 'ScaleBankRC':
            self.best_model = ScaleBankRC(args).float().to(args.device)

        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())

        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0



class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float().to(args.device)
            self.y_data = torch.from_numpy(y_train).long().to(args.device)
        else:
            self.x_data = X_train.float().to(args.device)
            self.y_data = y_train.long().to(args.device)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



@torch.no_grad()
def Test(monitor_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval()
    for data, label in monitor_loader:
        out = bestmodel(data, label)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    X = None
    if final == True:
        num_classes = args.num_class
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)
        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()
        per_class_acc = correct / total.clamp(min=1)
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item())
            # print(f"Class {i} Accuracy: {per_class_acc[i].item() * 100:.2f}%", '总数是:', int(total[i].item()))

    acc = int((predicts == labels).sum()) / len(labels)
    printtext = "ACC:{:.4f}".format(acc)
    print(printtext)
    return acc, X



@torch.no_grad()
def update_class_slot_memory(memory, model, train_loader):
    was_training = model.training
    model.eval()
    sums = torch.zeros_like(memory.prototypes)
    counts = torch.zeros(memory.num_classes, device=sums.device)
    for data, label in train_loader:
        _, auxiliary = model(data, label, return_aux=True)
        slot_features = auxiliary['slot_features']
        for class_tensor in label.unique():
            class_index = int(class_tensor.item())
            class_mask = label == class_tensor
            sums[class_index] += slot_features[class_mask].sum(dim=0)
            counts[class_index] += class_mask.sum()
    present_classes = counts > 0
    current_prototypes = sums / counts.clamp(min=1).view(-1, 1, 1)
    _record.memory_before(memory, current_prototypes, present_classes)
    memory.update(current_prototypes, present_classes)
    _record.memory_after(memory)
    model.train(was_training)



def Trainer(args, model):
    PredLossFun = nn.CrossEntropyLoss()
    class_slot_memory = ClassSlotRelationConsolidation(
        args.num_class, args.num_slots, args.slot_dim, args.class_slot_momentum
    ).to(args.device)
    seen_validation_loaders = []
    validation_matrix = []
    XX = []
    for task in range(args.tasks):
        _record.start(task)
        print('=' * 20, task + 1, '=' * 30)
        early_stopping = EarlyStopping(patience=15, verbose=True, dump=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3)
        train_loader, val_loader, monitor_loader = data_generator(args, task)
        seen_validation_loaders.append(monitor_loader)

        for epoch in range(1, args.epoches + 1):
            train_losses, val_losses, monitor_losses = [], [], []
            val_batch_sizes = []
            model.train()
            for data, label in train_loader:
                optimizer.zero_grad()
                out, auxiliary = model(data, label, return_aux=True)
                loss = PredLossFun(out, label) + args.slot_loss_weight * auxiliary['slot_loss']
                class_slot_loss = class_slot_memory(auxiliary['slot_features'], label)
                loss = loss + args.class_slot_loss_weight * class_slot_loss
                train_losses.append(loss.item())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite gradient')
                optimizer.step()

            with torch.no_grad():
                model.eval()
                for data, label in val_loader:
                    out = model(data, label)
                    loss = PredLossFun(out, label)
                    val_losses.append(loss.item())
                    val_batch_sizes.append(label.size(0))

            validation_loss = (float(np.average(val_losses, weights=val_batch_sizes))
                               if args.weighted_validation else float(np.mean(val_losses)))
            if not np.isfinite(validation_loss):
                raise FloatingPointError('Non-finite validation loss')

            with torch.no_grad():
                model.eval()
                for data, label in monitor_loader:
                    out= model(data, label)
                    loss = PredLossFun(out, label)
                    monitor_losses.append(loss.item())

            print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, validation_monitor_loss:{3:.5f}'.format(epoch,
                                                                                               np.mean(train_losses),
                                                                                               validation_loss,
                                                                                               np.mean(monitor_losses)))

            scheduler.step(validation_loss)
            early_stopping(validation_loss, model, epoch)
            _record.epoch(epoch, train_losses, val_losses, val_batch_sizes, validation_loss, early_stopping, optimizer)
            if early_stopping.early_stop:
                print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
                break
            if np.isnan(validation_loss) or np.isnan(np.mean(train_losses)):
                break

        update_class_slot_memory(class_slot_memory, model, train_loader)

        _record.models(model, early_stopping, args, epoch)
        accs = []
        for i, monitor_loader in enumerate(seen_validation_loaders):
            acc, xx = Test(monitor_loader, early_stopping.best_model, True)
            print(f"Accuracy on Domain {i + 1}: {acc:.4f}")
            accs.append(acc)
            if task == args.tasks - 1:
                XX.append(xx)
        validation_matrix.append(accs)
        _record.matrix(validation_matrix)
        print("  Accuracy Matrix:", flush=True)
        for row_index, row in enumerate(validation_matrix):
            row_str = "  ".join(f"{value * 100:.2f}" for value in row)
            print(f"  D{row_index + 1}: {row_str}", flush=True)


    print("\n=== Accuracy Matrix ===")
    for i, acc_row in enumerate(validation_matrix):
        row_str = ", ".join(f"{acc:.4f}" for acc in acc_row)
        print(f"After Domain {i + 1}: [{row_str}]")

    avg_acc = average_accuracy(validation_matrix)
    avg_forget = forgetting(validation_matrix)
    avg_pa_forget = performance_aware_forgetting(validation_matrix)
    XX = torch.tensor(XX)
    XX = torch.mean(XX, dim=0)
    print(f"\n=== Average Accuracy: {avg_acc:.4f} ===")
    print("Validation RF:", avg_forget)
    print("Validation PRF:", avg_pa_forget)

    print(XX.cpu().numpy())
    return avg_acc, avg_forget, avg_pa_forget, XX.cpu().numpy(), validation_matrix

