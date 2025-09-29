"""Training entry point for remote-sensing few-shot segmentation.

遥感小样本分割训练脚本入口点。
smoke run：
python training.py with \
        n_steps=2000 \
        max_iters_per_load=128 \
        scan_per_load=64 \
        save_snapshot_every=1000 \
        exp_prefix='stage2_smoke' \
        path.log_dir='./runs/stage2_smoke'
         
full run：

export STAGE1_ADAPTER_PATH=$(pwd)/runs/dense_openearthmap/adapter_lora_state.pt
export SUPPORT_TILE_FILE=$(pwd)/data/potsdam_bijie/splits/support_ids.txt
export STAGE2_TRAIN_QUERY_FILE=$(pwd)/data/potsdam_bijie/splits/train_query_ids.txt
export STAGE2_LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
python training.py with \
        n_steps=20000 \
        max_iters_per_load=256 \
        scan_per_load=128 \
        save_snapshot_every=2000 \
        exp_prefix='stage2_full' \
        task.n_shots=8 \
        path.log_dir='./runs/stage2_full'
        
python training.py with \
        model.adapter_state_path=/home/yukun/codes/paper5_dino-sat/DINOv3-based-Self-Supervised-Few-Shot-Disaster/runs/dense_openearthmap/adapter_lora_state.pt \
        model.lora=8 \
        support_id_whitelist@cfg=/home/yukun/codes/paper5_dino-sat/DINOv3-based-Self-Supervised-Few-Shot-Disaster/data/potsdam_bijie/splits/support_ids.txt \
        train_query_id_whitelist@cfg=/home/yukun/codes/paper5_dino-sat/DINOv3-based-Self-Supervised-Few-Shot-Disaster/data/potsdam_bijie/splits/train_query_ids.txt
"""

from __future__ import annotations

import os
import shutil
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
import dataloaders.augutils as myaug

from util.utils import set_seed, compose_wt_simple
from config_ssl_upload import ex


def get_train_transforms(_config):
    """根据配置构建训练阶段的数据增强流水线。"""

    # 使用自定义的增强模块，根据配置中指定的增强策略以及输入尺寸构建图像增强流程。
    return myaug.transform_with_label({"aug": myaug.get_aug(_config["which_aug"], _config["input_size"][0])})


def build_episode_loader(_config):
    """构建 episodic DataLoader，用于小样本分割训练。"""

    dataset_name = _config["dataset"]
    if dataset_name not in _config["path"]:
        raise KeyError(f"Missing path configuration for dataset '{dataset_name}'")

    dataset_act_labels = _config.get("dataset_act_labels", {})
    act_labels = dataset_act_labels.get(dataset_name)

    # 调用定制的数据加载器生成 episodic 数据集，episodes 为可迭代对象，dataset 为父级数据集对象。
    episodes, dataset = med_fewshot(
        dataset_name=dataset_name,
        base_dir=_config["path"][dataset_name]["data_dir"],
        idx_split=0,
        mode="train",
        scan_per_load=_config["scan_per_load"],
        transforms=get_train_transforms(_config),
        act_labels=act_labels,
        n_ways=_config["task"]["n_ways"],
        n_shots=_config["task"]["n_shots"],
        max_iters_per_load=_config["max_iters_per_load"],
        n_queries=_config["task"]["n_queries"],
        support_id_whitelist=_config.get("support_id_whitelist"),
        query_id_whitelist=_config.get("train_query_id_whitelist"),
        image_size=_config["input_size"][0],
    )
    # 构建 PyTorch DataLoader，负责按批次提供 episodic 数据。开启 shuffle 与 pinned memory 提升训练效率。
    loader = DataLoader(
        episodes,
        batch_size=_config["batch_size"],
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset


def compute_binary_iou(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """计算二类前景的 IoU 指标，用于监控模型分割性能。"""

    # 取 argmax 获得最终类别预测结果。
    preds = pred_logits.argmax(dim=1)
    # 构建前景布尔掩码，值为 1 代表前景，其余为背景。
    target_fg = target == 1
    pred_fg = preds == 1
    # 交集：预测与标注同时为前景的像素数；并集：预测或标注为前景的像素数。
    intersection = (pred_fg & target_fg).float().sum(dim=(-1, -2))
    union = (pred_fg | target_fg).float().sum(dim=(-1, -2)) + 1e-6
    # 对 batch 求平均，返回 python float。
    return (intersection / union).mean().item()


@ex.automain
def main(_run, _config, _log):
    """Sacred 自动调用的主函数，负责完整的训练流程。"""

    # 默认使用 float32 训练，可根据需要调整精度。
    precision = torch.float32
    # 如果存在可用 GPU，则优先使用 GPU 加速，否则退回 CPU。
    # 训练脚本在单卡场景下运行，如需多卡需额外修改模型和数据加载逻辑。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 固定随机数种子，确保实验结果可复现。
    set_seed(_config["seed"])

    # 若 Sacred 监控器存在，则整理源码快照，方便复现实验。
    if _run.observers:
        os.makedirs(f"{_run.observers[0].dir}/snapshots", exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            os.makedirs(os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"), exist_ok=True)
            _run.observers[0].save_file(source_file, f"source/{source_file}")
        shutil.rmtree(f"{_run.observers[0].basedir}/_sources")

    # TensorBoard 记录器：记录损失、IoU 等指标，便于可视化训练曲线。
    writer = SummaryWriter(f"{_run.observers[0].dir}/logs")

    _log.info("###### Create model ######")
    # 构建 FewShotSeg 模型，支持加载预训练权重；转换到目标设备并设定计算精度。
    model = FewShotSeg(
        image_size=_config["input_size"][0],
        pretrained_path=_config["reload_model_path"] or None,
        cfg=_config["model"],
    ).to(device, precision)
    model.train()

    _log.info("###### Build episodic loader ######")
    # 构建 episodic 训练数据加载器以及父数据集对象，用于后续可选的缓存刷新。
    trainloader, parent_dataset = build_episode_loader(_config)

    # 根据配置选择优化器类型，目前支持 SGD 与 AdamW。
    if _config["optim_type"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **_config["optim"])
    elif _config["optim_type"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=_config["lr"], eps=1e-5)
    else:
        raise NotImplementedError

    # 多阶段学习率调度器：在指定里程碑时刻乘以 gamma 衰减。
    scheduler = MultiStepLR(optimizer, milestones=_config["lr_milestones"], gamma=_config["lr_step_gamma"])

    # CrossEntropy 损失，支持忽略标签以及可选的类别权重（用于类别不平衡）。
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config["ignore_label"],
        weight=compose_wt_simple(_config["use_wce"], _config["dataset"]),
    )

    # 计算理论上需要的 sub-epoch 数量：取 max 以保证训练步数或 epoch 数不被截断。
    max_sub_epochs = max(1, _config["n_steps"] // _config["max_iters_per_load"], _config["epochs"])

    # i_iter: 全局已经执行的 iteration 数；losses_record: 用于缓存损失值（如需调试）。
    i_iter = 0
    losses_record = []

    for sub_epoch in range(max_sub_epochs):
        _log.info(f"###### Sub-epoch {sub_epoch + 1}/{max_sub_epochs} ######")
        # 部分数据集实现了 reload_buffer，用于周期性刷新支撑缓存。
        if hasattr(parent_dataset, "reload_buffer"):
            parent_dataset.reload_buffer()
            trainloader.dataset.update_index()

        # 使用 tqdm 创建进度条，实时显示损失与 IoU。
        pbar = tqdm(trainloader)
        # 梯度累积前需手动清零。
        optimizer.zero_grad()

        for idx, sample in enumerate(pbar):
            i_iter += 1
            # Support 图片按照 way -> shot 的结构存储，需要逐元素迁移到目标设备。
            support_images = [[shot.to(device, precision) for shot in way] for way in sample["support_images"]]
            # 前景/背景掩码同样需要迁移到设备，且转换为 float 以便后续计算。
            support_fg_mask = [
                [shot["fg_mask"].float().to(device, precision) for shot in way] for way in sample["support_mask"]
            ]
            support_bg_mask = [
                [shot["bg_mask"].float().to(device, precision) for shot in way] for way in sample["support_mask"]
            ]
            # Query 图片直接堆叠为批次，标签在最后拼接成一个 tensor。
            query_images = [img.to(device, precision) for img in sample["query_images"]]
            query_labels = torch.cat([lb.long().to(device) for lb in sample["query_labels"]], dim=0)

            # 前向传播，返回 query 预测、对齐损失等项；训练阶段 isval=False。
            out = model(
                support_images,
                support_fg_mask,
                support_bg_mask,
                query_images,
                isval=False,
                val_wsize=None,
            )
            query_pred, align_loss, *_ = out

            # 总损失 = 主任务交叉熵 + 原型对齐损失，反向传播累积梯度。
            loss = criterion(query_pred.float(), query_labels) + align_loss
            loss.backward()

            # 梯度累积到指定步数后才执行优化器更新和学习率调度。
            if (idx + 1) % _config["grad_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # 计算二分类 IoU，作为监控指标。
            iou = compute_binary_iou(query_pred.detach(), query_labels.detach())
            losses_record.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

            # 将损失、对齐损失与 IoU 写入 TensorBoard。
            writer.add_scalar("train/loss", loss.item(), i_iter)
            writer.add_scalar("train/align_loss", align_loss.item(), i_iter)
            writer.add_scalar("train/iou", iou, i_iter)

            # 按配置定期保存模型快照，便于中断恢复或模型选择。
            if (i_iter + 1) % _config["save_snapshot_every"] == 0:
                _log.info("###### Taking snapshot ######")
                torch.save(
                    model.state_dict(),
                    os.path.join(_run.observers[0].dir, "snapshots", f"{i_iter + 1}.pth"),
                )

            # 达到配置的训练步数后提前结束当前 sub-epoch。
            if i_iter >= _config["n_steps"]:
                break
        if i_iter >= _config["n_steps"]:
            break

    _log.info("Training finished.")
    # 关闭 TensorBoard 记录器，防止资源泄漏。
    writer.close()
