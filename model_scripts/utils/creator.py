import json
from os import sep
from chemprop import data, models, nn
from chemprop.nn import RegressionFFN, EvidentialFFN
from torch.utils.data import WeightedRandomSampler, DataLoader
from chemprop.data.collate import collate_batch
from utils.MPNN_WeightDecay import MPNNWithWeightDecay
from utils.custom_metric_log import PerTaskMAE

def create_model(config_path, scaler, train_targets: list[str], test_targets: list[str], extra_dim: int | float, sep_val=True):
    if isinstance(config_path, str):
        with open(config_path, "r") as f:
            config = json.load(f)
        config = config["train_loop_config"]
    else:
        config = config_path

    batch_norm = True
    if sep_val:
        metrics = []
        for i, task in enumerate(test_targets):
            metrics.append(PerTaskMAE(i, alias=f"mae_{task}"))  
    else:
        metrics = [nn.metrics().MAE()]

    agg = nn.NormAggregation()

    # Initialise message passing layer
    mp = nn.BondMessagePassing(
        d_h=config["message_hidden_dim"],
        depth=config["depth"],
        bias=True
    )
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    # Initialise predictor layer
    task_weights = [1.018, 1.000, 1.364, 1.134, 2.377, 2.373, 3.939, 5.259, 23.099]
    task_weights += [1] * (len(train_targets) - len(task_weights))
    ffn = EvidentialFFN(
        task_weights=task_weights,
        input_dim=config["message_hidden_dim"] + extra_dim,
        hidden_dim=config["ffn_hidden_dim"],
        n_layers=config["ffn_num_layers"],
        n_tasks=len(train_targets),
        dropout=config["dropout"],
        output_transform=output_transform
    )
    # ffn = nn.EvidentialFFN(
    #     n_tasks=len(targets),
    #     task_weights=task_weights,
    #     output_transform=output_transform,
    #     input_dim=mp.output_dim,
    #     hidden_dim=args.ffn_hidden_dim,
    #     n_layers=args.ffn_num_layers,
    #     dropout=args.dropout,
    # )
    mpnn = MPNNWithWeightDecay(mp, agg, ffn, batch_norm, metrics,
                       init_lr=config["init_lr"], 
                       max_lr=config["max_lr"],#config["lr"]*10, 
                       final_lr=config["final_lr"],#config["lr"], 
                       warmup_epochs=config["warmup"],
                       weight_decay=config["weight_decay"])

    return mpnn

def create_loader(config_path, train_dset, val_dset, test_dset, weights=None):
    sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dset),
                replacement=True
            ) if weights is not None else None

    if isinstance(config_path, str):
        with open(config_path, "r") as f:
            config = json.load(f)
        config = config["train_loop_config"]
    else:
        config = config_path
 
    train_loader = DataLoader(
        train_dset, 
        batch_size=config["batch_size"], 
        shuffle=(sampler is None),
        collate_fn=collate_batch,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        collate_fn=collate_batch,
        # sampler=sampler
    )

    test_loader = DataLoader(
        test_dset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        collate_fn=collate_batch 
    ) if test_dset is not None else None

    return train_loader, val_loader, test_loader