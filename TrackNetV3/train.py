import os
import time
import argparse
import csv
import json
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Shuttlecock_Trajectory_Dataset
from evaluate import select_best_checkpoint_report, write_json_report, write_validation_artifacts
from finetune import append_history, compute_validation_metrics, load_finetune_state, save_training_config, TrainingStateError
from test import eval_tracknet, eval_inpaintnet
from utils.general import ResumeArgumentParser, get_model, to_img_format
from utils.metric import WBCELoss
from utils.visualize import plot_heatmap_pred_sample, plot_traj_pred_sample, write_to_tb

SELECTION_METRIC = 'p_at_20'
DETECTION_RATE_GUARD = 0.02


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet', choices=['TrackNet', 'InpaintNet'], help='model type')
    parser.add_argument('--dataset_root', type=str, default='data', help='root directory of the adapted dataset')
    parser.add_argument('--cache_root', type=str, default=None, help='directory for dataset cache npz files')
    parser.add_argument('--seq_len', type=int, default=8, help='sequence length of input')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='validation epochs without an accepted P@20 improvement; 0 disables')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size of training')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'Adadelta'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='', choices=['', 'StepLR'], help='learning rate scheduler')
    parser.add_argument('--bg_mode', type=str, default='', choices=['', 'subtract', 'subtract_concat', 'concat'], help='background mode')
    parser.add_argument('--alpha', type=float, default=-1, help='alpha of sample mixup, -1 means no mixup')
    parser.add_argument('--frame_alpha', type=float, default=-1, help='alpha of frame mixup, -1 means no mixup')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='ratio of random mask during training InpaintNet')
    parser.add_argument('--tolerance', type=float, default=4, help='difference tolerance of center distance between prediction and ground truth in input size')
    parser.add_argument('--pretrained_tracknet', type=str, default=None, help='checkpoint path for model-weight initialization only')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path for resuming optimizer, scheduler, epoch, and best metric state')
    parser.add_argument('--resume_training', action='store_true', default=False, help='resume training from the current checkpoint in save_dir')
    parser.add_argument('--baseline_validation_detection_rate', type=float, default=0.0, help='minimum detection-rate reference for best-checkpoint guard')
    parser.add_argument('--baseline-validation-report', type=str, default=None, help='validation metrics report for baseline checkpoint selection')
    parser.add_argument('--seed', type=int, default=13, help='random seed')
    parser.add_argument('--save_dir', type=str, default='exp', help='directory to save the checkpoints and prediction result')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser


def _save_checkpoint(path, *, epoch, best_metric, model, optimizer, scheduler, param_dict, adapted_data_manifest, history):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent, delete=False)
    temporary.close()
    try:
        torch.save(dict(epoch=epoch,
                        best_metric=best_metric,
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict() if scheduler is not None else None,
                        param_dict=param_dict,
                        adapted_data_manifest=adapted_data_manifest,
                        history=history),
                   temporary.name)
        os.replace(temporary.name, target)
    finally:
        _safe_remove(temporary.name)


def _write_history(save_dir, history):
    history_path = os.path.join(save_dir, 'training_history.json')
    _atomic_write_json(history_path, history)


def _write_training_status(save_dir, *, selection_metric, early_stopping_patience,
                           selection_split='validation',
                           last_completed_epoch=None, stopped_early=False,
                           patience_without_improvement=0, best_metric=None):
    path = Path(save_dir) / 'training_status.json'
    _atomic_write_json(path, {
        'selection_metric': selection_metric,
        'early_stopping_patience': early_stopping_patience,
        'selection_split': selection_split,
        'last_completed_epoch': last_completed_epoch,
        'stopped_early': stopped_early,
        'patience_without_improvement': patience_without_improvement,
        'best_metric': best_metric,
    })
    return path


def accepted_validation_improvement(metrics, best_metric, baseline_detection_rate,
                                    detection_rate_guard=DETECTION_RATE_GUARD):
    """Check a validation P@20 improvement without weakening the DR guard."""
    if float(metrics.get('detection_rate', 0.0)) < float(baseline_detection_rate) - detection_rate_guard:
        return False
    return best_metric is None or float(metrics.get(SELECTION_METRIC, 0.0)) > float(best_metric.get(SELECTION_METRIC, 0.0))


def update_early_stopping(patience_without_improvement, *, metrics, best_metric,
                          baseline_detection_rate, patience,
                          detection_rate_guard=DETECTION_RATE_GUARD):
    accepted = accepted_validation_improvement(
        metrics, best_metric, baseline_detection_rate, detection_rate_guard
    )
    count = 0 if accepted else patience_without_improvement + 1
    return accepted, count, patience > 0 and count >= patience


def _reports_dir(save_dir):
    save_path = os.path.abspath(save_dir)
    if os.path.basename(save_path) == 'checkpoints':
        return os.path.join(os.path.dirname(save_path), 'reports')
    return os.path.join(save_path, 'reports')


def _epoch_checkpoint_path(save_dir, model_name, epoch):
    return os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1:03d}.pt')


def _epoch_report_name(epoch):
    return f'epoch_{epoch + 1:03d}.json'


def presence_confusion_matrix(summary):
    """Return visibility-presence counts, separate from localization counts."""
    return {
        'TP': float(summary.get('TP', 0.0)) + float(summary.get('FP1', 0.0)),
        'FN': float(summary.get('FN', 0.0)),
        'FP': float(summary.get('FP2', 0.0)),
        'TN': float(summary.get('TN', 0.0)),
    }


def _epoch_reports_root(reports_dir):
    return Path(reports_dir) / 'epochs'


def _epoch_metric_reports(reports_dir):
    return sorted(
        path for path in _epoch_reports_root(reports_dir).glob('epoch_*.json')
        if re.fullmatch(r'epoch_\d+\.json', path.name)
    )


def _atomic_write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent, mode='w', encoding='utf-8', delete=False)
    try:
        with temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
        os.replace(temporary.name, target)
    finally:
        _safe_remove(temporary.name)


def _reset_training_outputs(save_dir, reports_dir):
    """Remove only generated training artifacts for a fresh run.

    Baseline/test reports and adapted data are intentionally outside this reset.
    """
    checkpoint_root = Path(save_dir)
    for pattern in ('TrackNet_epoch_*.pt', 'epoch_top_*.pt', 'TrackNet_best.pt', 'TrackNet_cur.pt'):
        for path in checkpoint_root.glob(pattern):
            _safe_remove(path)
    reports_root = Path(reports_dir)
    shutil.rmtree(reports_root / 'epochs', ignore_errors=True)
    shutil.rmtree(reports_root / 'plots', ignore_errors=True)
    for path in reports_root.glob('epoch-*-validation.json'):
        _safe_remove(path)
    for path in (reports_root / 'metrics_history.csv', reports_root / 'best_checkpoint.json'):
        _safe_remove(path)


def _history_rows(history):
    rows = []
    for item in history:
        metrics = item.get('metrics') or {}
        row = {'epoch': item.get('epoch', 0) + 1, 'train_loss': item.get('train_loss'), 'val_loss': item.get('val_loss')}
        row.update(metrics)
        rows.append(row)
    return rows


def write_metrics_history_csv(reports_dir, history):
    rows = _history_rows(history)
    path = Path(reports_dir) / 'metrics_history.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['epoch', 'train_loss', 'val_loss', 'p_at_10', 'p_at_20', 'p_at_40', 'detection_rate', 'precision', 'recall', 'f1', 'mean_error', 'median_error', 'false_positive_rate']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    return path


def _plot_training_reports(reports_dir, history):
    """Write optional progress plots; training remains usable without matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plots_dir = Path(reports_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = _history_rows(history)
    epochs = [row['epoch'] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    value = lambda row, key: row.get(key) if row.get(key) is not None else float('nan')
    axes[0].plot(epochs, [value(row, 'train_loss') for row in rows], label='train loss')
    axes[0].plot(epochs, [value(row, 'val_loss') for row in rows], label='val loss')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    for key in ('p_at_10', 'p_at_20', 'p_at_40', 'detection_rate', 'precision', 'recall', 'f1', 'mean_error', 'median_error'):
        axes[1].plot(epochs, [value(row, key) for row in rows], label=key)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('score')
    axes[1].legend(ncol=3)
    figure.savefig(plots_dir / 'metrics_evolution.png')
    plt.close(figure)

    report_rows = []
    for path in _epoch_metric_reports(reports_dir):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            metrics = payload.get('metrics', {})
            if not isinstance(metrics, dict):
                continue
            report_rows.append((path.stem.replace('epoch_', ''), metrics))
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    if report_rows:
        figure, axis = plt.subplots(figsize=(10, 5))
        x = list(range(len(report_rows)))
        axis.plot(x, [metrics.get('p_at_20', 0) for _, metrics in report_rows], marker='o', label='P@20')
        axis.plot(x, [metrics.get('detection_rate', 0) for _, metrics in report_rows], marker='o', label='Detection Rate')
        axis.set_xticks(x, [epoch for epoch, _ in report_rows])
        axis.set_xlabel('evaluated epoch')
        axis.set_ylabel('validation score')
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / 'checkpoint_comparison.png')
        plt.close(figure)
        for epoch, metrics in report_rows:
            confusion = metrics.get('presence_confusion_matrix') or {}
            if not confusion:
                report_path = _epoch_reports_root(reports_dir) / f'epoch_{int(epoch):03d}.json'
                confusion = json.loads(report_path.read_text(encoding='utf-8')).get('confusion_matrix', {})
            figure, axis = plt.subplots(figsize=(4, 4))
            matrix = [[confusion.get('TP', 0), confusion.get('FN', 0)], [confusion.get('FP', 0), confusion.get('TN', 0)]]
            image = axis.imshow(matrix, cmap='Blues')
            axis.set_xticks([0, 1], ['predicted present', 'predicted absent'])
            axis.set_yticks([0, 1], ['visible', 'invisible'])
            for row_index in range(2):
                for column_index in range(2):
                    axis.text(column_index, row_index, matrix[row_index][column_index], ha='center', va='center')
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(_epoch_reports_root(reports_dir) / f'epoch_{int(epoch):03d}_confusion_matrix.png')
            plt.close(figure)
    return [plots_dir / 'metrics_evolution.png', plots_dir / 'checkpoint_comparison.png']


def _epoch_artifact_sort_key(artifact):
    metrics = artifact['metrics']
    return (
        -float(metrics['p_at_20']),
        -float(metrics['detection_rate']),
        float(metrics.get('median_error', float('inf'))),
        -int(artifact.get('epoch', -1)),
    )


def _safe_remove(path):
    if not path:
        return
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()


def _require_artifact_under_root(path, root, *, label):
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f'{label} escapes the active run root: {resolved_path}') from exc
    return str(resolved_path)


def _artifact_roots(reports_dir):
    reports_root = Path(reports_dir).resolve()
    return reports_root, (reports_root.parent / 'checkpoints').resolve()


def load_epoch_artifacts(reports_dir, max_epoch=None):
    artifacts = []
    reports_root, checkpoints_root = _artifact_roots(reports_dir)
    if not reports_root.exists():
        return artifacts
    epoch_reports = _epoch_metric_reports(reports_dir)
    epoch_reports += [path for path in reports_root.glob('epoch-*-validation.json') if re.fullmatch(r'epoch-\d+-validation\.json', path.name)]
    for report_path in sorted(epoch_reports):
        try:
            payload = json.loads(report_path.read_text(encoding='utf-8'))
            checkpoint_path = payload.get('checkpoint')
            metrics = payload.get('metrics')
            if not isinstance(metrics, dict) or not checkpoint_path:
                continue
            epoch_token = (report_path.stem.split('_')[1] if report_path.parent.name == 'epochs' else report_path.stem.split('-')[1])
            epoch = int(epoch_token) - 1
        except (OSError, json.JSONDecodeError, AttributeError, IndexError, TypeError, ValueError):
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        safe_report_path = _require_artifact_under_root(report_path, reports_root, label='epoch validation report')
        if checkpoint_path:
            try:
                checkpoint_path = _require_artifact_under_root(checkpoint_path, checkpoints_root, label='epoch checkpoint')
            except ValueError:
                continue
        if not checkpoint_path or not Path(checkpoint_path).exists():
            continue
        artifacts.append(
            {
                'epoch': epoch,
                'checkpoint': checkpoint_path,
                'report': str(report_path),
                'metrics': metrics,
            }
        )
    return artifacts


def select_best_epoch_artifact(*, reports_dir, baseline_validation_report, detection_rate_guard=0.02, max_epoch=None):
    baseline_report = json.loads(Path(baseline_validation_report).read_text(encoding='utf-8'))
    if baseline_report.get('split') != 'validation':
        raise TrainingStateError('baseline validation report must use split=validation')

    artifacts = load_epoch_artifacts(reports_dir, max_epoch=max_epoch)
    if not artifacts:
        raise TrainingStateError('no validation checkpoint candidates available')

    candidate_reports = []
    artifacts_by_checkpoint = {}
    for artifact in artifacts:
        report = json.loads(Path(artifact['report']).read_text(encoding='utf-8'))
        if report.get('split') != 'validation':
            raise TrainingStateError('candidate validation report must use split=validation')
        candidate_reports.append(report)
        artifacts_by_checkpoint[str(Path(artifact['checkpoint']).resolve())] = artifact

    selection = select_best_checkpoint_report(
        baseline_report,
        candidate_reports,
        detection_rate_guard=detection_rate_guard,
    )
    selected_checkpoint = str(Path(selection['selected_checkpoint']).resolve())
    selected_artifact = dict(artifacts_by_checkpoint[selected_checkpoint])
    selected_artifact['selection_report'] = selection
    return selected_artifact


def persist_epoch_validation_artifacts(
    *,
    save_dir,
    reports_dir,
    model_name,
    epoch,
    dataset_root,
    evaluation_mode,
    summary,
    selection_metrics,
    best_metric,
    model,
    optimizer,
    scheduler,
    param_dict,
    adapted_data_manifest,
    history,
):
    checkpoint_path = _epoch_checkpoint_path(save_dir, model_name, epoch)
    _save_checkpoint(
        checkpoint_path,
        epoch=epoch,
        best_metric=best_metric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        param_dict=param_dict,
        adapted_data_manifest=adapted_data_manifest,
        history=history,
    )
    report_path = write_validation_artifacts(
        _epoch_reports_root(reports_dir),
        split='validation',
        checkpoint=checkpoint_path,
        dataset_root=dataset_root,
        evaluation_mode=evaluation_mode,
        summary=summary,
        output_name=_epoch_report_name(epoch),
    )
    report_payload = json.loads(Path(report_path).read_text(encoding='utf-8'))
    report_payload['epoch'] = epoch + 1
    report_payload['train_loss'] = history[-1].get('train_loss') if history else None
    report_payload['val_loss'] = history[-1].get('val_loss') if history else None
    report_payload['confusion_matrix'] = presence_confusion_matrix(summary)
    _atomic_write_json(report_path, report_payload)
    confusion_path = _epoch_reports_root(reports_dir) / f'epoch_{epoch + 1:03d}_confusion_matrix.json'
    _atomic_write_json(confusion_path, report_payload['confusion_matrix'])
    return {
        'epoch': epoch,
        'checkpoint': checkpoint_path,
        'report': str(report_path),
        'metrics': selection_metrics,
    }


def retain_top_epoch_artifacts(epoch_artifacts, *, keep=3, preserve=None):
    if keep < 0:
        raise ValueError('keep must be >= 0')
    preserve = set(preserve or set())
    canonical_reports = any(Path(artifact['report']).parent.name == 'epochs' for artifact in epoch_artifacts)
    valid_artifacts = [artifact for artifact in epoch_artifacts if Path(artifact.get('checkpoint', '')).is_file()]
    ranked = sorted(valid_artifacts, key=_epoch_artifact_sort_key)
    kept = ranked[:keep]
    keep_keys = {(artifact['checkpoint'], artifact['report']) for artifact in kept}
    for artifact in ranked[keep:]:
        artifact_key = (artifact['checkpoint'], artifact['report'])
        if artifact_key not in preserve:
            continue
        kept.append(artifact)
        keep_keys.add(artifact_key)
        if len(kept) <= keep:
            continue
        for candidate in reversed(kept):
            candidate_key = (candidate['checkpoint'], candidate['report'])
            if candidate_key in preserve or candidate_key == artifact_key:
                continue
            kept.remove(candidate)
            keep_keys.remove(candidate_key)
            break
    for artifact in valid_artifacts:
        artifact_key = (artifact['checkpoint'], artifact['report'])
        if artifact_key in keep_keys:
            continue
        try:
            safe_report_path = _require_artifact_under_root(artifact['report'], Path(artifact['report']).resolve().parent, label='epoch validation report')
            report_root = Path(safe_report_path).resolve().parent
            if report_root.name == 'epochs':
                report_root = report_root.parent
            safe_checkpoint_path = _require_artifact_under_root(artifact['checkpoint'], report_root.parent / 'checkpoints', label='epoch checkpoint')
        except ValueError:
            continue
        _safe_remove(safe_checkpoint_path)
        # Reports are durable evaluation history; pruning never removes them.
    if canonical_reports and kept:
        checkpoints_root = Path(kept[0]['checkpoint']).resolve().parent if kept else None
        if checkpoints_root:
            staged = []
            staging_failed = False
            for index, artifact in enumerate(sorted(kept, key=_epoch_artifact_sort_key), start=1):
                source = Path(artifact['checkpoint'])
                fd, temporary_name = tempfile.mkstemp(prefix='.epoch_top_', suffix='.tmp', dir=checkpoints_root)
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    shutil.copyfile(source, temporary)
                except OSError:
                    _safe_remove(temporary)
                    staging_failed = True
                    break
                staged.append((artifact, temporary))
            if staging_failed:
                for _, temporary in staged:
                    _safe_remove(temporary)
                return kept
            renamed = []
            for index, (artifact, temporary) in enumerate(staged, start=1):
                target = checkpoints_root / f'epoch_top_{index:03d}.pt'
                os.replace(temporary, target)
                updated = dict(artifact, checkpoint=str(target))
                report_path = Path(updated['report'])
                try:
                    payload = json.loads(report_path.read_text(encoding='utf-8'))
                    payload['checkpoint'] = str(target)
                    _atomic_write_json(report_path, payload)
                except (OSError, json.JSONDecodeError):
                    pass
                renamed.append(updated)
            for legacy_checkpoint in checkpoints_root.glob('TrackNet_epoch_*.pt'):
                _safe_remove(legacy_checkpoint)
            for old_top in checkpoints_root.glob('epoch_top_*.pt'):
                if old_top not in {Path(item['checkpoint']) for item in renamed}:
                    _safe_remove(old_top)
            return renamed
    return kept


def mixup(x, y, alpha=0.5):
    """Returns mixed inputs, pairs of targets.
    
        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            alpha (float): Alpha of beta distribution

        Returns:
            x_mix (torch.Tensor): Mixed input tensor
            y_mix (torch.Tensor): Mixed target tensor
    """

    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)

    return x_mix, y_mix

def get_random_mask(mask_size, mask_ratio):
    """ Generate random mask by binomial distribution.
        1 means masked, 0 means not.
    
        Args:
            mask_size (Tuple[int, int]): Mask size (N, L)
            mask_ratio (float): Ratio of masked area

        Returns:
            mask (torch.Tensor): Random mask tensor with shape (N, L, 1)
    """

    mask = np.random.binomial(1, mask_ratio, size=mask_size)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(-1)

    return mask

def train_tracknet(model, optimizer, data_loader, param_dict):
    """ Train TrackNet model for one epoch.

        Args:
            model (torch.nn.Module): TrackNet model
            optimizer (torch.optim): Optimizer
            data_loader (torch.utils.data.DataLoader): Data loader
            param_dict (Dict): Parameters
                - param_dict['alpha'] (float): Alpha of sample mixup
                - param_dict['verbose'] (bool): Control whether to show progress bar
                - param_dict['bg_mode'] (str): For visualizing current prediction
                - param_dict['save_dir'] (str): For saving current prediction
        
        Returns:
            (float): Average loss
    """

    model.train()
    epoch_loss = []

    if param_dict['verbose']:
        data_prob = tqdm(train_loader)
    else:
        data_prob = data_loader
    
    for step, (_, x, y, c, _) in enumerate(data_prob):
        optimizer.zero_grad()
        x, y = x.float().cuda(), y.float().cuda()

        # Sample mixup
        if param_dict['alpha'] > 0:
            x, y = mixup(x, y, param_dict['alpha'])
        
        y_pred = model(x)
        loss = WBCELoss(y_pred, y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if param_dict['verbose'] and (step + 1) % display_step == 0:
            data_prob.set_description(f'Training')
            data_prob.set_postfix(loss=loss.item())

        # Visualize current prediction
        if (step + 1) % display_step == 0:
            x, y, y_pred = x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            c = c.numpy()
            
            # Transform inputs to image format (N, L, H , W, C)
            if param_dict['bg_mode'] == 'subtract':
                x = to_img_format(x)
            elif param_dict['bg_mode'] == 'subtract_concat':
                x = to_img_format(x, num_ch=4)
            elif param_dict['bg_mode'] == 'concat':
                x = to_img_format(x, num_ch=3)
                x = x[:, 1:, :, :, :]
            else:
                x = to_img_format(x, num_ch=3)
            y = to_img_format(y)
            y_pred = to_img_format(y_pred)
            plot_heatmap_pred_sample(x[0], y[0], y_pred[0], c[0], bg_mode=param_dict['bg_mode'], save_dir=param_dict['save_dir'])
    
    return float(np.mean(epoch_loss))
   
def train_inpaintnet(model, optimizer, data_loader, param_dict):
    """ Train InpaintNet model for one epoch.

        Args:
            model (torch.nn.Module): InpaintNet model
            optimizer (torch.optim): Optimizer
            data_loader (torch.utils.data.DataLoader): Data loader
            param_dict (Dict): parameters
                - param_dict['verbose'] (bool): Control whether to show progress bar
                - param_dict['mask_ratio'] (float): Ratio of masked area
                - param_dict['save_dir'] (str): For saving current prediction

        Returns:
            (float): Average loss
    """

    model.train()
    epoch_loss = []

    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (_, coor_pred, coor_gt, _, vis_gt, _) in enumerate(data_prob):
        optimizer.zero_grad()
        coor_pred, coor_gt, vis_gt = coor_pred.float().cuda(), coor_gt.float().cuda(), vis_gt.float().cuda()

        # Sample random mask as inpainting mask
        mask = get_random_mask(mask_size=coor_gt.shape[:2], mask_ratio=param_dict['mask_ratio']).cuda() # (N, L, 1)
        inpaint_mask = torch.logical_and(vis_gt, mask).int() # visible and masked area
        
        coor_pred = coor_pred * (1 - inpaint_mask) # masked area is set to 0
        refine_coor = model(coor_pred, inpaint_mask)

        # Calculate masked loss
        masked_refine_coor = refine_coor * inpaint_mask
        masked_gt_coor = coor_gt * inpaint_mask
        loss = nn.MSELoss()(masked_refine_coor, masked_gt_coor)
        epoch_loss.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if param_dict['verbose'] and (step + 1) % display_step == 0:
            data_prob.set_description(f'Training')
            data_prob.set_postfix(loss=loss.item())

        # Visualize current prediction
        if (step + 1) % display_step == 0:
            coor_gt, refine_coor, inpaint_mask = coor_gt.detach().cpu().numpy(), refine_coor.detach().cpu().numpy(), inpaint_mask.detach().cpu().numpy()
            plot_traj_pred_sample(coor_gt[0], refine_coor[0], inpaint_mask[0], save_dir=param_dict['save_dir'])
    
    return float(np.mean(epoch_loss))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    param_dict = vars(args)
    param_dict['selection_metric'] = SELECTION_METRIC

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print(f"TensorBoard: start with 'tensorboard --logdir {os.path.join(args.save_dir, 'logs')}', view at http://localhost:6006/")
    tb_writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    display_step = 4 if args.debug else 100
    num_workers = args.batch_size if args.batch_size <= 16 else 16
    
    if args.resume_training and args.resume is None:
        args.resume = os.path.join(args.save_dir, f'{args.model_name}_cur.pt')
        param_dict['resume'] = args.resume

    if args.pretrained_tracknet and args.resume:
        raise SystemExit('pretrained_tracknet and resume are mutually exclusive')
    if args.model_name == 'TrackNet' and not args.baseline_validation_report:
        raise SystemExit('baseline-validation-report is required for TrackNet best-checkpoint selection')

    if args.early_stopping_patience < 0:
        raise SystemExit('early_stopping_patience must be >= 0')

    reports_dir = _reports_dir(args.save_dir)
    if args.resume is None:
        _reset_training_outputs(args.save_dir, reports_dir)

    training_metadata = {
        'early_stopping_patience': args.early_stopping_patience,
        'selection_metric': SELECTION_METRIC,
        'selection_split': 'validation',
    }

    cache_root = args.cache_root or os.path.join(args.save_dir, 'cache')
    config_path = save_training_config(
        os.path.join(args.save_dir, 'training_config.json'),
        {
            'model_name': args.model_name,
            'dataset_root': args.dataset_root,
            'cache_root': cache_root,
            'seq_len': args.seq_len,
            'epochs': args.epochs,
            **training_metadata,
            'bg_mode': args.bg_mode,
            'pretrained_tracknet': args.pretrained_tracknet,
            'resume_checkpoint': args.resume,
            'baseline_validation_detection_rate': args.baseline_validation_detection_rate,
            'baseline_validation_report': args.baseline_validation_report,
            'save_dir': args.save_dir,
        },
    )
    _write_training_status(args.save_dir, **training_metadata)
    manifest_path = os.path.join(args.dataset_root, 'manifest.json')
    manifest_fingerprint = None
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest_fingerprint = json.load(handle).get('fingerprint')

    print(f'Parameters: {param_dict}')
    print(f'Load dataset...')
    data_mode = 'heatmap' if args.model_name == 'TrackNet' else 'coordinate'
    train_dataset = Shuttlecock_Trajectory_Dataset(root_dir=args.dataset_root, cache_root=cache_root, split='train', seq_len=args.seq_len, sliding_step=1, data_mode=data_mode, bg_mode=args.bg_mode, frame_alpha=args.frame_alpha, debug=args.debug)
    val_dataset = Shuttlecock_Trajectory_Dataset(root_dir=args.dataset_root, cache_root=cache_root, split='val', seq_len=args.seq_len, sliding_step=args.seq_len, data_mode=data_mode, bg_mode=args.bg_mode, debug=args.debug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    print(f'Create {args.model_name}...')
    model = get_model(args.model_name, args.seq_len, args.bg_mode).cuda() if args.model_name == 'TrackNet' else get_model(args.model_name).cuda()
    train_fn = train_tracknet if args.model_name == 'TrackNet' else train_inpaintnet
    eval_fn = eval_tracknet if args.model_name == 'TrackNet' else eval_inpaintnet

    # Create optimizer
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('Invalid optimizer.')

    # Create lr scheduler
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.1)
    else:
        scheduler = None

    state = load_finetune_state(
        model_name=args.model_name,
        seq_len=args.seq_len,
        bg_mode=args.bg_mode,
        dataset_root=args.dataset_root,
        save_dir=args.save_dir,
        adapted_data_manifest=manifest_fingerprint,
        pretrained_tracknet=args.pretrained_tracknet,
        resume_checkpoint=args.resume,
        device='cpu',
    )
    if state['model_state_dict'] is not None:
        model.load_state_dict(state['model_state_dict'], strict=False)
    if state['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and state['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])

    best_metric = state['best_metric']
    history = state['history']
    start_epoch = state['start_epoch']
    baseline_detection_rate = 0.0
    if args.model_name == 'TrackNet':
        with open(args.baseline_validation_report, 'r', encoding='utf-8') as handle:
            baseline_detection_rate = json.load(handle)['metrics']['detection_rate']
    patience_without_improvement = 0
    if state['mode'] == 'resume':
        print(f'Resume training from epoch {start_epoch}...')
    elif state['mode'] == 'pretrained':
        print(f'Loaded pretrained weights from {args.pretrained_tracknet} and started a fresh run...')


    print(f'Start training...')
    train_start_time = time.time()
    epoch_artifacts = load_epoch_artifacts(reports_dir, max_epoch=start_epoch - 1)
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch [{epoch+1} / {args.epochs}]')
        start_time = time.time()
        train_loss = train_fn(model, optimizer, train_loader, param_dict)
        val_loss, val_res = eval_fn(model, val_loader, param_dict)
        selection_metrics = compute_validation_metrics(summary=val_res) if args.model_name == 'TrackNet' else None
        history = append_history(history, epoch=epoch, train_loss=train_loss, val_loss=val_loss, metrics=selection_metrics or val_res, metadata=training_metadata)
        _write_history(args.save_dir, history)
        if args.model_name == 'TrackNet':
            new_artifact = persist_epoch_validation_artifacts(
                    save_dir=args.save_dir,
                    reports_dir=reports_dir,
                    model_name=args.model_name,
                    epoch=epoch,
                    dataset_root=args.dataset_root,
                    evaluation_mode='weight',
                    summary=val_res,
                    selection_metrics=selection_metrics,
                    best_metric=best_metric,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    param_dict=param_dict,
                    adapted_data_manifest=manifest_fingerprint,
                    history=history,
                )
            epoch_artifacts = [artifact for artifact in epoch_artifacts if artifact.get('epoch') != epoch]
            epoch_artifacts.append(new_artifact)
            write_metrics_history_csv(reports_dir, history)
            _plot_training_reports(reports_dir, history)
        write_to_tb(args.model_name, tb_writer, (train_loss, val_loss), val_res, epoch)

        if args.lr_scheduler:
            scheduler.step()
        
        # Pick best model
        save_best = False
        previous_best_metric = best_metric
        accepted_improvement = False
        if args.model_name == 'TrackNet':
            try:
                selected_artifact = select_best_epoch_artifact(
                    reports_dir=reports_dir,
                    baseline_validation_report=args.baseline_validation_report,
                    max_epoch=epoch,
                )
                epoch_artifacts = retain_top_epoch_artifacts(
                    epoch_artifacts,
                    keep=3,
                    preserve={(selected_artifact['checkpoint'], selected_artifact['report'])},
                )
                save_best = int(selected_artifact.get('epoch', -1)) == epoch
                best_metric = {'epoch': selected_artifact['epoch'], **selected_artifact['metrics']}
            except TrainingStateError as exc:
                print(f'Skip best checkpoint update: {exc}')
        else:
            cur_val_acc = val_res['inpaint']['accuracy']
            if best_metric is None or cur_val_acc >= best_metric.get('accuracy', 0.):
                best_metric = {'epoch': epoch, 'accuracy': cur_val_acc}
                save_best = True

        if save_best:
            _save_checkpoint(os.path.join(args.save_dir, f'{args.model_name}_best.pt'),
                             epoch=epoch,
                             best_metric=best_metric,
                             model=model,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             param_dict=param_dict,
                             adapted_data_manifest=manifest_fingerprint,
                             history=history)
            if args.model_name == 'TrackNet':
                write_json_report(
                    os.path.join(reports_dir, 'best_checkpoint.json'),
                    {
                        'checkpoint': os.path.join(args.save_dir, f'{args.model_name}_best.pt'),
                        'dataset_root': args.dataset_root,
                        'selection_metric': 'p_at_20',
                        'split': 'validation',
                        'metrics': best_metric,
                    },
                )
        
        # Save current model
        _save_checkpoint(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'),
                         epoch=epoch,
                         best_metric=best_metric,
                         model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         param_dict=param_dict,
                         adapted_data_manifest=manifest_fingerprint,
                         history=history)

        if args.model_name == 'TrackNet':
            accepted_improvement, patience_without_improvement, should_stop = update_early_stopping(
                patience_without_improvement,
                metrics=selection_metrics,
                best_metric=previous_best_metric,
                baseline_detection_rate=baseline_detection_rate,
                patience=args.early_stopping_patience,
            )
            _write_training_status(
                args.save_dir,
                **training_metadata,
                last_completed_epoch=epoch + 1,
                stopped_early=should_stop,
                patience_without_improvement=patience_without_improvement,
                best_metric=best_metric,
            )
            if should_stop:
                print(f'Early stopping after epoch {epoch + 1}: no accepted validation P@20 improvement for {args.early_stopping_patience} epochs.')
                break
        
        print(f'Epoch runtime: {(time.time() - start_time) / 3600.:.2f} hrs')
    
    tb_writer.close()
    print(f'Training config saved to {config_path}')
    print(f'Training time: {(time.time() - train_start_time) / 3600.:.2f} hrs')
    print('Done......')
