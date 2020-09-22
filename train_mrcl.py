import os
import argparse
import socket
import time
import sys

import pprint as pp
from comet_ml import Experiment
import higher
from tqdm import tqdm

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter

import numpy as np
from eval.meta_eval import meta_test


def parse_option():

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument(
        "--eval_freq", type=int, default=1000, help="meta-eval frequency"
    )
    parser.add_argument(
        "--print_freq", type=int, default=50, help="meta-eval frequency"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--apply_every", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num of workers to use"
    )
    parser.add_argument(
        "--num_steps", type=int, default=20000, help="number of training steps"
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--inner_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--num_inner_steps", type=int, default=5)
    parser.add_argument("--num_inner_steps_test", type=int, default=10)
    # parser.add_argument(
    #     "--test_inner_lr", type=float, default=0.1, help="learning rate"
    # )

    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        nargs="*",
        default=[],
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.5, help="decay rate for learning rate"
    )

    # dataset
    parser.add_argument("--model", type=str, default="resnet12", choices=model_pool)
    parser.add_argument(
        "--dataset",
        type=str,
        default="miniImageNet",
        choices=["miniImageNet", "tieredImageNet", "CIFAR-FS", "FC100"],
    )
    parser.add_argument("--transform", type=str, default="A", choices=transforms_list)
    parser.add_argument("--use_trainval", action="store_true", help="use trainval set")

    # cosine annealing
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument("--cosine_factor", type=float, default=0.05)

    # specify folder
    parser.add_argument("--model_path", type=str, default="", help="path to save model")
    parser.add_argument("--tb_path", type=str, default="", help="path to tensorboard")
    parser.add_argument("--data_root", type=str, default="", help="path to data root")

    # meta setting
    parser.add_argument(
        "--n_test_runs", type=int, default=300, metavar="N", help="Number of test runs"
    )
    parser.add_argument(
        "--n_ways",
        type=int,
        default=5,
        metavar="N",
        help="Number of classes for doing each classification run",
    )
    parser.add_argument(
        "--n_shots", type=int, default=5, metavar="N", help="Number of shots in test"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=5,
        metavar="N",
        help="Number of queries in test",
    )
    parser.add_argument(
        "--n_qry_way", type=int, default=64, metavar="N", help="Number of query in test"
    )
    parser.add_argument(
        "--n_qry_shot", type=int, default=2, metavar="N", help="Number of query in test"
    )
    parser.add_argument(
        "--n_aug_support_samples",
        default=0,
        type=int,
        help="The number of augmented samples for each meta test sample",
    )
    parser.add_argument(
        "--n_aug_qry_samples",
        default=0,
        type=int,
        help="The number of augmented samples for each meta test sample",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        metavar="test_batch_size",
        help="Size of test batch)",
    )

    parser.add_argument(
        "-t", "--trial", type=str, default="1", help="the experiment id"
    )
    parser.add_argument("--logcomet", action="store_true", default=False)
    parser.add_argument("--comet_project_name", default="debug", type=str)
    parser.add_argument("--comet_workspace", default="debug", type=str)
    parser.add_argument(
        "--no_dropblock", default=True, action="store_false", dest="dropblock"
    )
    parser.add_argument("--drop_rate", default=0.1, type=float)
    parser.add_argument(
        "--initializer",
        default="kaiming_normal",
        choices=["glorot_uniform", "kaiming_normal"],
    )
    parser.add_argument("--track_stats", default=False, action="store_true")
    parser.add_argument("--reset_head", choices=["zero", "kaiming", "glorot"])
    parser.add_argument("--weight_norm", default=0, choices=[0, 1], type=int)
    parser.add_argument(
        "--activation", default="leaky_relu", choices=["relu", "leaky_relu"]
    )
    parser.add_argument("--normalization", default="bn", choices=["bn", "affine"])

    parser.add_argument(
        "--no_aug_keep_original",
        default=True,
        action="store_false",
        dest="aug_keep_original",
    )

    parser.add_argument(
        "--augment",
        default="none",
        choices=[
            "none",
            "all",
            "qry",
            "spt",
        ],
    )

    opt = parser.parse_args()
    # opt.dropblock = not opt.no_dropblock
    opt.data_aug = False  # hard code

    # if opt.dataset == "CIFAR-FS" or opt.dataset == "FC100":
    #     opt.transform = "D"

    if opt.use_trainval:
        opt.trial = opt.trial + "_trainval"

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = "./models_pretrained"
    if not opt.tb_path:
        opt.tb_path = "./tensorboard"
    if not opt.data_root:
        opt.data_root = "./data/{}".format(opt.dataset)
    # else:
    #     opt.data_root = "{}/{}".format(opt.data_root, opt.dataset)

    # iterations = opt.lr_decay_epochs.split(",")
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))
    print("lr_decay_epochs")
    print(opt.lr_decay_epochs)

    # opt.model_name = "{}_{}_lr_{}_decay_{}_trans_{}_bsz_{}".format(
    opt.model_name = "mrcl_{}_{}_olr_{}_ilr_{}_{}w_{}s_{}qs_aug_{}_bsz_{}_reset_{}_{}_{}_{}".format(
        opt.model,
        opt.dataset,
        opt.learning_rate,
        opt.inner_lr,
        opt.n_ways,
        opt.n_shots,
        opt.n_queries,
        # opt.weight_decay,
        opt.augment,
        opt.batch_size,
        opt.reset_head,
        opt.initializer,
        opt.activation,
        # opt.epochs,
        opt.normalization,
    )

    if opt.weight_norm == 1:
        opt.model_name = "{}_wn".format(opt.model_name)

    if opt.cosine:
        opt.model_name = "{}_cosine_{}".format(opt.model_name, opt.cosine_factor)

    # if opt.adam:
    #     opt.model_name = "{}_useAdam".format(opt.model_name)

    if opt.model == "resnet12":
        if opt.dropblock and opt.drop_rate > 0:
            opt.model_name = "{}_dropblock{}".format(opt.model_name, opt.drop_rate)
        elif opt.drop_rate > 0:
            opt.model_name = "{}_dropout{}".format(opt.model_name, opt.drop_rate)
        else:
            opt.model_name = "{}_no-drop".format(opt.model_name)

    opt.model_name = "{}_trial_{}".format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def main():
    opt = parse_option()

    print(pp.pformat(vars(opt)))

    train_partition = "trainval" if opt.use_trainval else "train"
    if opt.dataset == "miniImageNet":
        train_trans, test_trans = transforms_options[opt.transform]

        if opt.augment == "none":
            train_train_trans = train_test_trans = test_trans
        elif opt.augment == "all":
            train_train_trans = train_test_trans = train_trans
        elif opt.augment == "spt":
            train_train_trans = train_trans
            train_test_trans = test_trans
        elif opt.augment == "qry":
            train_train_trans = test_trans
            train_test_trans = train_trans

        print("spt trans")
        print(train_train_trans)
        print("qry trans")
        print(train_test_trans)

        sub_batch_size, rmd = divmod(opt.batch_size, opt.apply_every)
        assert rmd == 0
        print("Train sub batch-size:", sub_batch_size)

        meta_train_dataset = MetaImageNet(
            args=opt,
            partition="train",
            train_transform=train_train_trans,
            test_transform=train_test_trans,
            fname="miniImageNet_category_split_train_phase_%s.pickle",
            fix_seed=False,
            n_test_runs=10000000,  # big number to never stop
            new_labels=False,
        )
        meta_trainloader = DataLoader(
            meta_train_dataset,
            batch_size=sub_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers,
        )
        meta_train_dataset_qry = MetaImageNet(
            args=opt,
            partition="train",
            train_transform=train_train_trans,
            test_transform=train_test_trans,
            fname="miniImageNet_category_split_train_phase_%s.pickle",
            fix_seed=False,
            n_test_runs=10000000,  # big number to never stop
            new_labels=False,
            n_ways=opt.n_qry_way,
            n_shots=opt.n_qry_shot,
            n_queries=0,
        )
        meta_trainloader_qry = DataLoader(
            meta_train_dataset_qry,
            batch_size=sub_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers,
        )
        meta_val_dataset = MetaImageNet(
            args=opt,
            partition="val",
            train_transform=test_trans,
            test_transform=test_trans,
            fix_seed=False,
            n_test_runs=200,
            n_ways=5,
            n_shots=5,
            n_queries=15,
        )
        meta_valloader = DataLoader(
            meta_val_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
        )
        # if opt.use_trainval:
        #     n_cls = 80
        # else:
        #     n_cls = 64
        n_cls = len(meta_train_dataset.classes)

    print(n_cls)

    # x_spt, y_spt, x_qry, y_qry = next(iter(meta_trainloader))

    # x_spt2, y_spt2, x_qry2, y_qry2 = next(iter(meta_trainloader_qry))

    # print(x_spt, y_spt, x_qry, y_qry)
    # print(x_spt2, y_spt2, x_qry2, y_qry2)
    # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
    # print(x_spt2.shape, y_spt2.shape, x_qry2.shape, y_qry2.shape)

    model = create_model(
        opt.model,
        n_cls,
        opt.dataset,
        opt.drop_rate,
        opt.dropblock,
        opt.track_stats,
        opt.initializer,
        opt.weight_norm,
        activation=opt.activation,
        normalization=opt.normalization,
    )

    print(model)

    # criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
        device = torch.device("cuda")
        # if opt.n_gpu > 1:
        #     model = nn.DataParallel(model)
        model = model.to(device)
        criterion = criterion.to(device)
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print("Learning rate")
    print(opt.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
    )
    # classifier = model.classifier()
    print("Inner Learning rate")
    print(opt.inner_lr)
    inner_opt = torch.optim.SGD(
        model.classifier.parameters(),
        lr=opt.inner_lr,
    )
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    comet_logger = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name=opt.comet_project_name,
        workspace=opt.comet_workspace,
        disabled=not opt.logcomet,
        auto_metric_logging=False,
    )
    comet_logger.set_name(opt.model_name)
    comet_logger.log_parameters(vars(opt))
    comet_logger.set_model_graph(str(model))

    if opt.cosine:
        eta_min = opt.learning_rate * opt.cosine_factor
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, opt.num_steps, eta_min, -1
        )

    # routine: supervised pre-training
    data_sampler = iter(meta_trainloader)
    data_sampler_qry = iter(meta_trainloader_qry)
    pbar = tqdm(
        range(1, opt.num_steps + 1),
        miniters=opt.print_freq,
        mininterval=5,
        maxinterval=30,
        ncols=0,
    )
    best_val_acc = 0.0
    for step in pbar:

        if not opt.cosine:
            adjust_learning_rate(step, opt, optimizer)
        # print("==> training...")

        time1 = time.time()

        foa = 0.0
        fol = 0.0
        ioa = 0.0
        iil = 0.0
        fil = 0.0
        iia = 0.0
        fia = 0.0
        for j in range(opt.apply_every):

            x_spt, y_spt, x_qry, y_qry = [t.to(device) for t in next(data_sampler)]
            x_qry2, y_qry2, _, _ = [t.to(device) for t in next(data_sampler_qry)]
            y_spt = y_spt.flatten(1)
            y_qry2 = y_qry2.flatten(1)

            x_qry = torch.cat((x_spt, x_qry, x_qry2), 1)
            y_qry = torch.cat((y_spt, y_qry, y_qry2), 1)

            if step == 1 and j == 0:
                print(x_spt.size(), y_spt.size(), x_qry.size(), y_qry.size())

            info = train_step(
                model,
                model.classifier,
                None,
                # inner_opt,
                opt.inner_lr,
                x_spt,
                y_spt,
                x_qry,
                y_qry,
                reset_head=opt.reset_head,
                num_steps=opt.num_inner_steps,
            )

            _foa = info["foa"] / opt.batch_size
            _fol = info["fol"] / opt.batch_size
            _ioa = info["ioa"] / opt.batch_size
            _iil = info["iil"] / opt.batch_size
            _fil = info["fil"] / opt.batch_size
            _iia = info["iia"] / opt.batch_size
            _fia = info["fia"] / opt.batch_size

            _fol.backward()

            foa += _foa.detach()
            fol += _fol.detach()
            ioa += _ioa.detach()
            iil += _iil.detach()
            fil += _fil.detach()
            iia += _iia.detach()
            fia += _fia.detach()

        optimizer.step()
        optimizer.zero_grad()

        if opt.cosine:
            scheduler.step()
        if (step == 1) or (step % opt.eval_freq == 0):
            val_info = test_run(
                iter(meta_valloader),
                model,
                model.classifier,
                inner_opt,
                num_inner_steps=opt.num_inner_steps_test,
                device=device,
            )
            val_acc_feat, val_std_feat = meta_test(
                model, meta_valloader, use_logit=False
            )

            val_acc = val_info["outer"]["acc"].cpu()
            val_loss = val_info["outer"]["loss"].cpu()

            print(f"\nValidation step {step}")
            print(f"MAML 5-way-5-shot accuracy: {val_acc.item()}")
            print(f"LR 5-way-5-shot accuracy: {val_acc_feat}+-{val_std_feat}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(
                    f"New best validation accuracy {val_acc.item()} saving checkpoints\n"
                )
                # print(val_acc.item())

                torch.save(
                    {
                        "opt": opt,
                        "model": model.state_dict()
                        if opt.n_gpu <= 1
                        else model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "val_acc_lr": val_acc_feat,
                    },
                    os.path.join(opt.save_folder, "{}_best.pth".format(opt.model)),
                )

            comet_logger.log_metrics(
                dict(
                    fol=val_loss,
                    foa=val_acc,
                    acc_lr=val_acc_feat,
                ),
                step=step,
                prefix="val",
            )

            logger.log_value("val_acc", val_acc, step)
            logger.log_value("val_loss", val_loss, step)
            logger.log_value("val_acc_lr", val_acc_feat, step)

        if (step == 1) or (step % opt.eval_freq == 0) or (step % opt.print_freq == 0):

            tfol = fol.cpu()
            tfoa = foa.cpu()
            tioa = ioa.cpu()
            tiil = iil.cpu()
            tfil = fil.cpu()
            tiia = iia.cpu()
            tfia = fia.cpu()

            comet_logger.log_metrics(
                dict(
                    fol=tfol,
                    foa=tfoa,
                    ioa=tfoa,
                    iil=tiil,
                    fil=tfil,
                    iia=tiia,
                    fia=tfia,
                ),
                step=step,
                prefix="train",
            )

            logger.log_value("train_acc", info["foa"], step)
            logger.log_value("train_loss", info["fol"], step)

            pbar.set_postfix(
                # iol=f"{info['iol'].item():.2f}",
                fol=f"{tfol.item():.2f}",
                # ioa=f"{info['ioa'].item():.2f}",
                foa=f"{tfoa.item():.2f}",
                ioa=f"{tioa.item():.2f}",
                iia=f"{tiia.item():.2f}",
                fia=f"{tfia.item():.2f}",
                vl=f"{val_loss.item():.2f}",
                va=f"{val_acc.item():.2f}",
                valr=f"{val_acc_feat:.2f}",
                # iil=f"{info['iil'].item():.2f}",
                # fil=f"{info['fil'].item():.2f}",
                # iia=f"{info['iia'].item():.2f}",
                # fia=f"{info['fia'].item():.2f}",
                # counter=info["counter"],
                refresh=True,
            )

    # save the last model
    state = {
        "opt": opt,
        "model": model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    save_file = os.path.join(opt.save_folder, "{}_last.pth".format(opt.model))
    torch.save(state, save_file)


class MetaLearner(nn.Module):
    def __init__(self, encoder, classifier, inner_opt):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.inner_opt = inner_opt
        # self.outer_opt = outer_opt

    def forward(
        self,
        x_spt,
        y_spt,
        x_qry,
        y_qry,
        reset_head="none",
        cls_indexes=None,
        num_steps=5,
    ):
        # inner_opt = torch.optim.SGD(
        #     self.classifier.parameters(),
        #     lr=self.inner_lr,
        # )
        return train_step(
            self.encoder,
            self.classifier,
            None,
            self.inner_opt,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            reset_head,
            cls_indexes,
            num_steps,
        )


def train_step(
    model,
    classifier,
    outer_opt,
    inner_opt_lr,
    x_spt,
    y_spt,
    x_qry,
    y_qry,
    reset_head="none",
    cls_indexes=None,
    num_steps=5,
):
    # if reset_head == "zero":
    # print("Resetting head")
    # nn.init.zeros_(classifier.weight)
    # print(classifier.weight)

    #  outer_opt.zero_grad()

    # total_loss = 0.
    counter = 0
    iia = 0.0  # initial inner accuracy
    fia = 0.0
    iil = 0.0
    fil = 0.0
    ioa = 0.0
    foa = 0.0
    iol = 0.0
    fol = 0.0  # final outer loss
    for task_num in range(x_spt.size(0)):

        # print(task_num)
        # print(fol)
        # print(foa)
        # print(fia)
        # print(fil)

        inner_opt = torch.optim.SGD(classifier.parameters(), lr=inner_opt_lr)

        with higher.innerloop_ctx(
            classifier, inner_opt, copy_initial_weights=False,
        ) as (fmodel, diffopt):


            cls_idxs = torch.unique(y_spt[task_num])

            for cls_idx in cls_idxs:
                if reset_head == "zero":
                    # print("resetting zero")
                    torch.nn.init.zeros_(fmodel._parameters["weight"][cls_idx].unsqueeze(0))
                elif reset_head == "kaiming":
                    torch.nn.init.kaiming_normal_(fmodel._parameters["weight"][cls_idx].unsqueeze(0))
                elif reset_head == "glorot":
                    torch.nn.init.xavier_uniform_(fmodel._parameters["weight"][cls_idx].unsqueeze(0))
                else:
                    raise NameError(f"Reset head {reset_head} unknown")

            # print(fmodel)
            # print(cls_idxs)
            # print(fmodel._parameters["weight"])
            # print(fmodel._parameters["weight"].size())
            # print(fmodel._parameters["weight"][cls_idxs])
            # print(fmodel._parameters["weight"][cls_idxs].size())
            # print(y_spt)

            # sys.exit()

            features, _ = model(x_spt[task_num], is_feat=True)
            feat = features[-1]

            loss_history = []
            acc_history = []

            for _ in range(num_steps):
                logits = fmodel(feat)
                loss = nn.functional.cross_entropy(logits, y_spt[task_num])
                diffopt.step(loss)

                loss_history.append(loss)
                acc_history.append(
                    (logits.argmax(-1) == y_spt[task_num]).float().mean()
                )

            logits = fmodel(feat)
            _fia = (logits.argmax(-1) == y_spt[task_num]).float().mean()

            loss_history.append(nn.functional.cross_entropy(logits, y_spt[task_num]))
            acc_history.append(fia)

            features, _ = model(x_qry[task_num], is_feat=True)

            logits = fmodel(features[-1])
            _fol = nn.functional.cross_entropy(logits, y_qry[task_num])
            _foa = (logits.argmax(-1) == y_qry[task_num]).float().mean()

            logits = fmodel(features[-1], params=fmodel.parameters(time=0))
            _iol = nn.functional.cross_entropy(logits, y_qry[task_num])
            _ioa = (logits.argmax(-1) == y_qry[task_num]).float().mean()

            fol += _fol

            # with torch.no_grad():
            ioa += _ioa
            foa += _foa
            iol += _iol
            iil += loss_history[0]
            fil += loss_history[-1]
            iia += acc_history[0]
            fia += _fia

        counter += 1

    # iia = iia / counter
    # fia = fia / counter
    # iil = iil / counter
    # fil = fil / counter
    # ioa = ioa / counter
    # foa = foa / counter
    # iol = iol / counter
    # fol = fol / counter

    # fol.backward()
    # outer_opt.step()
    # outer_opt.zero_grad()

    return dict(
        iia=iia,  # .detach(),
        fia=fia,  # .detach(),
        iil=iil,  # .detach(),
        fil=fil,  # .detach(),
        ioa=ioa,  # .detach(),
        foa=foa,  # .detach(),
        iol=iol,  # .detach(),
        fol=fol,  # .detach(),
        # counter=counter,
    )


def test_run(loader, encoder, classifier, inner_opt, num_inner_steps=10, device=None):
    encoder.eval()
    classifier.eval()

    ia = 0.0
    il = 0.0
    oa = 0.0
    ol = 0.0
    counter = 0
    for i in range(len(loader)):
        x_spt, y_spt, x_qry, y_qry = [t[0].to(device) for t in next(loader)]
        y_spt = y_spt.flatten()

        with higher.innerloop_ctx(classifier, inner_opt, track_higher_grads=False) as (
            fmodel,
            diffopt,
        ):
            torch.nn.init.zeros_(fmodel._parameters["weight"])
            torch.nn.init.zeros_(fmodel._parameters["bias"])
            with torch.no_grad():
                features, _ = encoder(x_spt, is_feat=True)
            feat = features[-1]

            # loss_history = []
            # acc_history = []

            with torch.set_grad_enabled(True):
                for _ in range(num_inner_steps):
                    logits = fmodel(feat)
                    loss = nn.functional.cross_entropy(logits, y_spt)
                    diffopt.step(loss)

                # loss_history.append(loss)
                # acc_history.append((logits.argmax(-1) == y_spt).float().mean())

            with torch.no_grad():
                logits = fmodel(feat)
                fia = (logits.argmax(-1) == y_spt).float().mean()

                #  loss_history.append(nn.functional.cross_entropy(logits, y_spt))
                # acc_history.append(fia)

                features, _ = encoder(x_qry, is_feat=True)
                # iol = nn.functional.cross_entropy(logits, y_qry)
                # ioa = (logits.argmax(-1) == y_qry).float().mean()

                logits = fmodel(features[-1])
                fol = nn.functional.cross_entropy(logits, y_qry)
                foa = (logits.argmax(-1) == y_qry).float().mean()

                counter += 1
                ia += fia
                il += loss
                oa += foa
                ol += fol

    return dict(
        inner=dict(acc=ia / counter, loss=il / counter),
        outer=dict(acc=oa / counter, loss=ol / counter),
    )


if __name__ == "__main__":
    main()