'''
by lyuwenyu
modified by ChatGPT for PCB baseline paper-style outputs
'''
import time
import json
import csv
import datetime
from pathlib import Path
from collections import OrderedDict, defaultdict

import torch

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    """
    Enhanced detection solver.

    Compared with the original det_solver.py, this version additionally saves
    paper-friendly outputs in one training run:
      - checkpoints/best.pth
      - checkpoints/last.pth
      - logs/metrics.csv
      - logs/metrics.jsonl
      - curves/loss_curve.png
      - curves/map_curve.png
      - curves/precision_recall_curve.png
      - analysis/final_val_metrics.json
      - analysis/per_class_ap_val.csv / .png
      - analysis/confusion_matrix_val.csv / .png
      - visualizations/val_best/*.jpg, if image path can be resolved
      - final test outputs, if cfg.test_dataloader exists
    """

    # You can tune these if needed.
    CONFUSION_IOU_THR = 0.50
    CONFUSION_SCORE_THR = 0.25
    VIS_SCORE_THR = 0.30
    VIS_MAX_IMAGES = 30
    VIS_MAX_DETS_PER_IMAGE = 50

    def fit(self):
        print("Start training")
        self.train()

        args = self.cfg

        # Prepare optional test dataloader if the config defines it.
        self.test_dataloader = None
        if getattr(self.cfg, 'test_dataloader', None) is not None:
            self.test_dataloader = dist.warp_loader(
                self.cfg.test_dataloader,
                shuffle=self.cfg.test_dataloader.shuffle,
            )

        self._setup_paper_output_dirs()
        self._save_run_config_summary()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        self._save_model_info(n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        best_stat = {
            'epoch': -1,
            'mAP50': -1.0,
            'mAP50_95': -1.0,
        }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler
            )

            self.lr_scheduler.step()

            # Save checkpoints.
            if self.output_dir:
                self._save_training_checkpoints(epoch)

            # Evaluate on validation set every epoch.
            module = self.ema.module if self.ema else self.model
            val_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            metric_dict = self._extract_coco_metrics(val_stats)
            val_mAP50 = float(metric_dict.get('mAP50', -1.0))
            val_mAP50_95 = float(metric_dict.get('mAP50_95', -1.0))

            # Save best checkpoints.
            if self.output_dir and dist.is_main_process():
                if val_mAP50 > best_stat['mAP50']:
                    best_stat['mAP50'] = val_mAP50
                    best_stat['mAP50_95'] = val_mAP50_95
                    best_stat['epoch'] = epoch
                    # Only one best checkpoint is saved, selected by validation mAP50.
                    dist.save_on_master(self.state_dict(epoch), self.paper_dirs['checkpoints'] / 'best.pth')
                    self._save_json(
                        {
                            'best_type': 'val_mAP50',
                            'epoch': epoch,
                            'epoch_readable': epoch + 1,
                            'mAP50': val_mAP50,
                            'mAP50_95': val_mAP50_95,
                        },
                        self.paper_dirs['logs'] / 'best_metrics.json'
                    )

            print('best_stat: ', best_stat)

            # Original-style log line, but rename validation metrics as val_* instead of test_*.
            log_stats = {
                **{f'train_{k}': self._to_builtin(v) for k, v in train_stats.items()},
                **{f'val_{k}': self._to_builtin(v) for k, v in val_stats.items()},
                **{f'val_{k}': self._to_builtin(v) for k, v in metric_dict.items()},
                'epoch': epoch,
                'epoch_readable': epoch + 1,
                'n_parameters': n_parameters,
            }

            if self.output_dir and dist.is_main_process():
                # Keep original log.txt behavior for compatibility.
                with (self.output_dir / 'log.txt').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats, ensure_ascii=False) + '\n')

                # Paper-friendly metrics.
                self._append_metrics(log_stats)
                self._plot_training_curves()


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if self.output_dir and dist.is_main_process():
            self._save_json(
                {
                    'training_time_seconds': int(total_time),
                    'training_time': total_time_str,
                    'best_stat': best_stat,
                },
                self.paper_dirs['logs'] / 'training_summary.json'
            )

        # Automatically run final paper-style evaluation with best checkpoint.
        self._run_final_paper_evaluation()

    def val(self):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        val_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, base_ds, self.device, self.output_dir
        )

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval['bbox'].eval, self.output_dir / 'eval.pth')

        return

    # ------------------------------------------------------------------
    # Output directory helpers
    # ------------------------------------------------------------------
    def _setup_paper_output_dirs(self):
        self.paper_dirs = {
            'checkpoints': self.output_dir / 'checkpoints',
            'logs': self.output_dir / 'logs',
            'curves': self.output_dir / 'curves',
            'analysis': self.output_dir / 'analysis',
            'visualizations': self.output_dir / 'visualizations',
            'speed': self.output_dir / 'speed',
        }
        if dist.is_main_process():
            for p in self.paper_dirs.values():
                p.mkdir(parents=True, exist_ok=True)

        self.metrics_rows = []

    def _save_run_config_summary(self):
        if not dist.is_main_process():
            return
        summary = {
            'output_dir': str(self.output_dir),
            'epoches': getattr(self.cfg, 'epoches', None),
            'device': str(getattr(self.cfg, 'device', None)),
            'note': 'This file is auto-generated by modified det_solver.py for paper-style experiment tracking.',
        }
        self._save_json(summary, self.paper_dirs['logs'] / 'run_config_summary.json')

    def _save_model_info(self, n_parameters):
        if not dist.is_main_process():
            return
        path = self.paper_dirs['speed'] / 'model_info.txt'
        with path.open('w', encoding='utf-8') as f:
            f.write(f'n_parameters: {n_parameters}\n')
            f.write(f'n_parameters_million: {n_parameters / 1e6:.4f}\n')
            f.write(f'device: {self.device}\n')

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _save_training_checkpoints(self, epoch):
        # Only save the latest checkpoint as last.pth. It is overwritten every epoch.
        # No periodic checkpoint and no root checkpoint.pth will be generated.
        dist.save_on_master(self.state_dict(epoch), self.paper_dirs['checkpoints'] / 'last.pth')

    # ------------------------------------------------------------------
    # Metric extraction and logging
    # ------------------------------------------------------------------
    @staticmethod
    def _to_builtin(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
        if np is not None and isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [DetSolver._to_builtin(v) for v in x]
        if isinstance(x, dict):
            return {k: DetSolver._to_builtin(v) for k, v in x.items()}
        try:
            if np is not None and isinstance(x, np.generic):
                return x.item()
        except Exception:
            pass
        return x

    def _extract_coco_metrics(self, stats):
        """
        Convert COCO bbox stats into named metrics.
        Typical COCO stats:
          [0] AP@[.50:.95]
          [1] AP@.50
          [2] AP@.75
          [3] AP small
          [4] AP medium
          [5] AP large
          [6] AR maxDets=1
          [7] AR maxDets=10
          [8] AR maxDets=100
        """
        out = OrderedDict()
        if 'coco_eval_bbox' in stats:
            values = stats['coco_eval_bbox']
            values = self._to_builtin(values)
            if isinstance(values, (list, tuple)) and len(values) >= 6:
                out['mAP50_95'] = float(values[0])
                out['mAP50'] = float(values[1])
                out['mAP75'] = float(values[2])
                out['AP_small'] = float(values[3])
                out['AP_medium'] = float(values[4])
                out['AP_large'] = float(values[5])
                if len(values) >= 9:
                    out['AR_1'] = float(values[6])
                    out['AR_10'] = float(values[7])
                    out['AR_100'] = float(values[8])
        return out

    def _append_metrics(self, row):
        self.metrics_rows.append(row)

        jsonl_path = self.paper_dirs['logs'] / 'metrics.jsonl'
        with jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

        # Re-write CSV with union of keys so new keys are preserved.
        keys = []
        for r in self.metrics_rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)

        csv_path = self.paper_dirs['logs'] / 'metrics.csv'
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.metrics_rows:
                writer.writerow({k: self._csv_safe(r.get(k, '')) for k in keys})

    @staticmethod
    def _csv_safe(v):
        if isinstance(v, (list, tuple, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    def _plot_training_curves(self):
        if plt is None or not self.metrics_rows:
            return

        epochs = [r.get('epoch_readable', r.get('epoch', 0)) for r in self.metrics_rows]

        def collect(key):
            y = []
            for r in self.metrics_rows:
                v = r.get(key, None)
                try:
                    y.append(float(v))
                except Exception:
                    y.append(None)
            return y

        def plot_one(keys, title, ylabel, filename):
            has_data = False
            plt.figure()
            for key in keys:
                y = collect(key)
                if any(v is not None for v in y):
                    xs = [e for e, v in zip(epochs, y) if v is not None]
                    ys = [v for v in y if v is not None]
                    if xs and ys:
                        plt.plot(xs, ys, label=key)
                        has_data = True
            if not has_data:
                plt.close()
                return
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.paper_dirs['curves'] / filename, dpi=300)
            plt.close()

        plot_one(
            ['train_loss', 'train_loss_vfl', 'train_loss_bbox', 'train_loss_giou'],
            'Training Loss Curves',
            'Loss',
            'loss_curve.png'
        )
        plot_one(
            ['val_mAP50', 'val_mAP50_95', 'val_mAP75'],
            'Validation mAP Curves',
            'mAP',
            'map_curve.png'
        )
        plot_one(
            ['val_AR_1', 'val_AR_10', 'val_AR_100'],
            'Validation Recall Curves',
            'AR',
            'recall_curve.png'
        )
        plot_one(
            ['train_lr', 'lr'],
            'Learning Rate Curve',
            'Learning Rate',
            'lr_curve.png'
        )

    # ------------------------------------------------------------------
    # Final evaluation and analysis
    # ------------------------------------------------------------------
    def _run_final_paper_evaluation(self):
        if not self.output_dir or not dist.is_main_process():
            return

        best_path = self.paper_dirs['checkpoints'] / 'best.pth'
        if not best_path.exists():
            print(f'[PaperOutput] best checkpoint not found: {best_path}. Skip final paper evaluation.')
            return

        print(f'[PaperOutput] Loading best checkpoint for final evaluation: {best_path}')
        state = torch.load(best_path, map_location='cpu')
        self.load_state_dict(state)

        # Final validation-set analysis.
        val_base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        self._final_eval_one_split('val', self.val_dataloader, val_base_ds)

        # Optional final test-set analysis.
        if getattr(self, 'test_dataloader', None) is not None:
            test_base_ds = get_coco_api_from_dataset(self.test_dataloader.dataset)
            self._final_eval_one_split('test', self.test_dataloader, test_base_ds)
        else:
            self._save_json(
                {
                    'message': 'cfg.test_dataloader is not defined, so final test evaluation was skipped.',
                    'suggestion': 'Add test_dataloader to pcb_detection.yml if you want automatic final test metrics.',
                },
                self.paper_dirs['analysis'] / 'final_test_skipped.json'
            )

    def _final_eval_one_split(self, split_name, dataloader, base_ds):
        print(f'[PaperOutput] Final evaluation on {split_name} set...')
        module = self.ema.module if self.ema else self.model
        final_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, dataloader, base_ds, self.device, self.output_dir
        )

        metrics = self._extract_coco_metrics(final_stats)
        final_payload = {
            'split': split_name,
            'metrics': metrics,
            'raw_stats': self._to_builtin(final_stats),
        }
        self._save_json(final_payload, self.paper_dirs['analysis'] / f'final_{split_name}_metrics.json')
        self._save_dict_csv(metrics, self.paper_dirs['analysis'] / f'final_{split_name}_metrics.csv')

        if coco_evaluator is None or 'bbox' not in coco_evaluator.coco_eval:
            return

        coco_eval = coco_evaluator.coco_eval['bbox']
        self._save_per_class_ap(coco_eval, split_name)
        self._save_confusion_matrix(coco_eval, split_name)
        self._save_dataset_distribution(coco_eval.cocoGt, split_name)
        self._save_detection_visualizations(coco_eval, dataloader, split_name)

    def _save_per_class_ap(self, coco_eval, split_name):
        if np is None:
            return
        if not hasattr(coco_eval, 'eval') or coco_eval.eval is None:
            return
        if 'precision' not in coco_eval.eval:
            return

        precision = coco_eval.eval['precision']  # [T, R, K, A, M]
        iou_thrs = coco_eval.params.iouThrs
        cat_ids = list(coco_eval.params.catIds)
        area_index = 0  # all area
        maxdet_index = -1

        def mean_valid(arr):
            arr = arr[arr > -1]
            if arr.size == 0:
                return float('nan')
            return float(np.mean(arr))

        # IoU=0.50 index
        iou50_index = int(np.argmin(np.abs(iou_thrs - 0.50)))

        rows = []
        for k, cat_id in enumerate(cat_ids):
            cat_info = coco_eval.cocoGt.cats.get(cat_id, {})
            cat_name = cat_info.get('name', str(cat_id))
            ap50 = mean_valid(precision[iou50_index, :, k, area_index, maxdet_index])
            ap5095 = mean_valid(precision[:, :, k, area_index, maxdet_index])
            rows.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'AP50': ap50,
                'AP50_95': ap5095,
            })

        csv_path = self.paper_dirs['analysis'] / f'per_class_ap_{split_name}.csv'
        self._write_rows_csv(rows, csv_path)

        if plt is not None and rows:
            names = [r['category_name'] for r in rows]
            ap50s = [r['AP50'] for r in rows]
            ap5095s = [r['AP50_95'] for r in rows]

            plt.figure(figsize=(max(8, len(rows) * 1.2), 5))
            x = np.arange(len(rows))
            width = 0.35
            plt.bar(x - width / 2, ap50s, width, label='AP50')
            plt.bar(x + width / 2, ap5095s, width, label='AP50:95')
            plt.xticks(x, names, rotation=30, ha='right')
            plt.ylabel('AP')
            plt.ylim(0, 1.05)
            plt.title(f'Per-class AP on {split_name}')
            plt.grid(axis='y', linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.paper_dirs['analysis'] / f'per_class_ap_{split_name}.png', dpi=300)
            plt.close()

    def _save_confusion_matrix(self, coco_eval, split_name):
        if np is None:
            return

        coco_gt = coco_eval.cocoGt
        coco_dt = coco_eval.cocoDt
        cat_ids = list(coco_eval.params.catIds)
        img_ids = list(coco_eval.params.imgIds)
        cat_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        names = [coco_gt.cats.get(cid, {}).get('name', str(cid)) for cid in cat_ids]
        bg_index = len(cat_ids)
        matrix = np.zeros((len(cat_ids) + 1, len(cat_ids) + 1), dtype=np.int64)

        for img_id in img_ids:
            gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], iscrowd=None)
            gt_anns = [a for a in coco_gt.loadAnns(gt_ann_ids) if a.get('iscrowd', 0) == 0]
            dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)
            dt_anns = [d for d in dt_anns if float(d.get('score', 0.0)) >= self.CONFUSION_SCORE_THR]
            dt_anns = sorted(dt_anns, key=lambda x: float(x.get('score', 0.0)), reverse=True)

            matched_gt = set()
            for det in dt_anns:
                det_box = det['bbox']
                pred_cat = det['category_id']
                pred_idx = cat_to_index.get(pred_cat, bg_index)

                best_iou = 0.0
                best_gt_i = None
                for gi, gt in enumerate(gt_anns):
                    if gi in matched_gt:
                        continue
                    iou = self._bbox_iou_xywh(det_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_i = gi

                if best_gt_i is not None and best_iou >= self.CONFUSION_IOU_THR:
                    matched_gt.add(best_gt_i)
                    gt_cat = gt_anns[best_gt_i]['category_id']
                    gt_idx = cat_to_index.get(gt_cat, bg_index)
                    matrix[gt_idx, pred_idx] += 1
                else:
                    # False positive: background predicted as this class.
                    matrix[bg_index, pred_idx] += 1

            for gi, gt in enumerate(gt_anns):
                if gi not in matched_gt:
                    gt_idx = cat_to_index.get(gt['category_id'], bg_index)
                    # False negative: this class predicted as background.
                    matrix[gt_idx, bg_index] += 1

        labels = names + ['background']
        rows = []
        for i, row_name in enumerate(labels):
            row = {'gt/pred': row_name}
            for j, col_name in enumerate(labels):
                row[col_name] = int(matrix[i, j])
            rows.append(row)
        self._write_rows_csv(rows, self.paper_dirs['analysis'] / f'confusion_matrix_{split_name}.csv')

        if plt is not None:
            plt.figure(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.7)))
            plt.imshow(matrix, interpolation='nearest')
            plt.title(f'Confusion Matrix on {split_name}')
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45, ha='right')
            plt.yticks(tick_marks, labels)
            thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    plt.text(j, i, str(matrix[i, j]), ha='center', va='center',
                             color='white' if matrix[i, j] > thresh else 'black')
            plt.ylabel('Ground Truth')
            plt.xlabel('Prediction')
            plt.tight_layout()
            plt.savefig(self.paper_dirs['analysis'] / f'confusion_matrix_{split_name}.png', dpi=300)
            plt.close()

    def _save_dataset_distribution(self, coco_gt, split_name):
        cat_ids = sorted(list(coco_gt.cats.keys()))
        rows = []
        wh_rows = []
        center_rows = []

        for cat_id in cat_ids:
            ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
            anns = coco_gt.loadAnns(ann_ids)
            cat_name = coco_gt.cats.get(cat_id, {}).get('name', str(cat_id))
            rows.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'num_boxes': len(anns),
            })
            for a in anns:
                x, y, w, h = a['bbox']
                wh_rows.append({
                    'category_id': cat_id,
                    'category_name': cat_name,
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'aspect_ratio': w / h if h else 0,
                })
                img_info = coco_gt.imgs.get(a['image_id'], {})
                img_w = img_info.get('width', None)
                img_h = img_info.get('height', None)
                center_rows.append({
                    'category_id': cat_id,
                    'category_name': cat_name,
                    'cx_norm': (x + w / 2) / img_w if img_w else '',
                    'cy_norm': (y + h / 2) / img_h if img_h else '',
                })

        self._write_rows_csv(rows, self.paper_dirs['analysis'] / f'dataset_category_distribution_{split_name}.csv')
        self._write_rows_csv(wh_rows, self.paper_dirs['analysis'] / f'bbox_distribution_{split_name}.csv')
        self._write_rows_csv(center_rows, self.paper_dirs['analysis'] / f'bbox_center_distribution_{split_name}.csv')

        if plt is not None and rows:
            names = [r['category_name'] for r in rows]
            counts = [r['num_boxes'] for r in rows]
            plt.figure(figsize=(max(8, len(rows) * 1.2), 5))
            plt.bar(names, counts)
            plt.xticks(rotation=30, ha='right')
            plt.ylabel('Number of boxes')
            plt.title(f'Category Distribution on {split_name}')
            plt.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(self.paper_dirs['analysis'] / f'dataset_category_distribution_{split_name}.png', dpi=300)
            plt.close()

        if plt is not None and wh_rows:
            plt.figure(figsize=(6, 5))
            plt.scatter([r['width'] for r in wh_rows], [r['height'] for r in wh_rows], s=8, alpha=0.6)
            plt.xlabel('Box width')
            plt.ylabel('Box height')
            plt.title(f'BBox Width-Height Distribution on {split_name}')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(self.paper_dirs['analysis'] / f'bbox_wh_distribution_{split_name}.png', dpi=300)
            plt.close()

    def _save_detection_visualizations(self, coco_eval, dataloader, split_name):
        if Image is None or ImageDraw is None:
            return

        img_root = self._resolve_image_root_from_dataset(dataloader.dataset)
        if img_root is None:
            self._save_json(
                {'message': 'Cannot resolve image root from dataset, visualization skipped.'},
                self.paper_dirs['visualizations'] / f'{split_name}_visualization_skipped.json'
            )
            return

        split_dir = self.paper_dirs['visualizations'] / f'{split_name}_best'
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_gt = coco_eval.cocoGt
        coco_dt = coco_eval.cocoDt
        img_ids = list(coco_eval.params.imgIds)[:self.VIS_MAX_IMAGES]

        for img_id in img_ids:
            img_info = coco_gt.imgs.get(img_id, {})
            file_name = img_info.get('file_name', None)
            if not file_name:
                continue
            img_path = img_root / file_name
            if not img_path.exists():
                # Sometimes file_name may include subdirectory.
                alt = Path(file_name)
                if alt.exists():
                    img_path = alt
                else:
                    continue

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            draw = ImageDraw.Draw(image)

            # Draw GT boxes in thin lines.
            gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(gt_ids)
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                cat_name = coco_gt.cats.get(ann['category_id'], {}).get('name', str(ann['category_id']))
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=1)
                draw.text((x, max(0, y - 10)), f'GT:{cat_name}', fill=(0, 255, 0))

            # Draw predictions.
            dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(dt_ids)
            dt_anns = [d for d in dt_anns if float(d.get('score', 0.0)) >= self.VIS_SCORE_THR]
            dt_anns = sorted(dt_anns, key=lambda x: float(x.get('score', 0.0)), reverse=True)[:self.VIS_MAX_DETS_PER_IMAGE]
            for det in dt_anns:
                x, y, w, h = det['bbox']
                score = float(det.get('score', 0.0))
                cat_name = coco_gt.cats.get(det['category_id'], {}).get('name', str(det['category_id']))
                draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
                draw.text((x, y + h + 2), f'{cat_name}:{score:.2f}', fill=(255, 0, 0))

            out_name = Path(file_name).name
            image.save(split_dir / out_name)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_iou_xywh(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter_w = max(0.0, xb - xa)
        inter_h = max(0.0, yb - ya)
        inter = inter_w * inter_h
        area1 = max(0.0, w1) * max(0.0, h1)
        area2 = max(0.0, w2) * max(0.0, h2)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _resolve_image_root_from_dataset(self, dataset):
        # Unwrap common dataset wrappers.
        visited = set()
        cur = dataset
        while id(cur) not in visited:
            visited.add(id(cur))
            for attr in ['img_folder', 'root', 'img_root', 'img_dir']:
                if hasattr(cur, attr):
                    value = getattr(cur, attr)
                    if value:
                        return Path(value)
            if hasattr(cur, 'dataset'):
                cur = cur.dataset
            else:
                break
        return None

    @staticmethod
    def _save_json(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(DetSolver._to_builtin(data), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_dict_csv(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in data.items():
                writer.writerow([k, v])

    @staticmethod
    def _write_rows_csv(rows, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with path.open('w', encoding='utf-8-sig') as f:
                f.write('')
            return
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: DetSolver._csv_safe(r.get(k, '')) for k in keys})
