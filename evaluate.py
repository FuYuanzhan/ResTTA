"""Evaluation of reconstructed PET volumes.

Protocol (identical to the paper's ``our_paper_metric.py``):

* load ``*_result.img`` (prediction) and ``*_hr.img`` (ground truth) with
  ``medpy.io.load``;
* zero out voxels whose ground-truth intensity is ``<= 0.05`` (background);
* compute PSNR / NMSE on the non-zero region, with
  ``data_range = max(pred, gt) - min(pred, gt)``;
* compute SSIM on the full volume.

Usage:
    python evaluate.py --result_dir outputs/dose_1/tta
    python evaluate.py --root outputs --doses 1,5,10 --arms baseline,tta
"""

import argparse
import json
import os

import numpy as np

BACKGROUND_THRESHOLD = 0.05


def _load_metric_libs():
    from medpy.io import load as medpy_load
    from skimage.metrics import normalized_root_mse as ski_nmse
    from skimage.metrics import peak_signal_noise_ratio as ski_psnr
    from skimage.metrics import structural_similarity as ski_ssim
    return medpy_load, ski_psnr, ski_ssim, ski_nmse


def evaluate_results_dir(result_dir, pred_suffix="_result.img",
                         gt_suffix="_hr.img",
                         background_threshold=BACKGROUND_THRESHOLD):
    """Evaluate one directory of ``*_result.img`` / ``*_hr.img`` pairs.

    Returns:
        ``(avg_psnr, avg_ssim, avg_nmse, per_subject)`` where ``per_subject``
        maps ``subject_id -> (psnr, ssim, nmse)``.
    """
    medpy_load, ski_psnr, ski_ssim, ski_nmse = _load_metric_libs()

    gt_paths, pred_paths = [], []
    for root, _, files in sorted(os.walk(result_dir)):
        for f in files:
            if f.endswith(gt_suffix):
                gt_paths.append(os.path.join(root, f))
            if f.endswith(pred_suffix):
                pred_paths.append(os.path.join(root, f))

    if not gt_paths:
        print(f"  [WARNING] no {gt_suffix} / {pred_suffix} found in {result_dir}")
        return 0.0, 0.0, 0.0, {}

    gt_paths.sort()
    pred_paths.sort()
    if len(gt_paths) != len(pred_paths):
        raise RuntimeError(
            f"Unequal number of predictions/GT in {result_dir}: "
            f"{len(pred_paths)} vs {len(gt_paths)}")

    per_subject = {}
    total_psnr, total_ssim, total_nmse = [], [], []

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        print(f"  {pred_path}")
        pred, _ = medpy_load(pred_path)
        gt, _ = medpy_load(gt_path)
        if pred.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: {pred_path} {pred.shape} "
                               f"vs {gt_path} {gt.shape}")

        background = gt <= background_threshold
        gt = gt.copy()
        gt[background] = 0
        pred = pred.copy()
        pred[background] = 0

        idx = np.nonzero(gt)
        pred_vals = pred[idx]
        gt_vals = gt[idx]
        data_range = (max(pred_vals.max(), gt_vals.max()) -
                      min(pred_vals.min(), gt_vals.min()))

        cur_psnr = ski_psnr(pred_vals, gt_vals, data_range=data_range)
        cur_ssim = ski_ssim(pred, gt, multi_channel=1, data_range=data_range)
        cur_nmse = ski_nmse(pred_vals, gt_vals) ** 2

        print(f"  PSNR: {cur_psnr:.6f} SSIM: {cur_ssim:.6f} "
              f"NMSE: {cur_nmse:.6f}")

        total_psnr.append(cur_psnr)
        total_ssim.append(cur_ssim)
        total_nmse.append(cur_nmse)

        subject_id = os.path.basename(pred_path)[:-len(pred_suffix)]
        per_subject[subject_id] = (cur_psnr, cur_ssim, cur_nmse)

    return (float(np.mean(total_psnr)), float(np.mean(total_ssim)),
            float(np.mean(total_nmse)), per_subject)


def write_per_subject_csv(csv_path, per_subject):
    with open(csv_path, "w") as f:
        f.write("subject_id,psnr,ssim,nmse\n")
        for k in sorted(per_subject):
            p, s, n = per_subject[k]
            f.write(f"{k},{p:.6f},{s:.6f},{n:.6f}\n")


def metrics_mean_std(per_subject):
    ps = [v[0] for v in per_subject.values()]
    ss = [v[1] for v in per_subject.values()]
    ns = [v[2] for v in per_subject.values()]
    return ((float(np.mean(ps)), float(np.std(ps))),
            (float(np.mean(ss)), float(np.std(ss))),
            (float(np.mean(ns)), float(np.std(ns))))


def evaluate_root(root, doses=None, arms=("baseline", "tta"), **kwargs):
    """Evaluate ``root/dose_*/{arm}`` and print/save a comparison table."""
    results = {}
    for dose in doses or sorted(
            [d[len("dose_"):] for d in os.listdir(root)
             if d.startswith("dose_")], key=lambda x: int(x)):
        dose_dir = os.path.join(root, f"dose_{dose}")
        row = {}
        for arm in arms:
            arm_dir = os.path.join(dose_dir, arm)
            if not os.path.isdir(arm_dir):
                continue
            print(f"\n--- dose={dose} arm={arm} ---")
            psnr, ssim, nmse, per_subject = evaluate_results_dir(
                arm_dir, **kwargs)
            row[arm] = {"psnr": psnr, "ssim": ssim, "nmse": nmse,
                        "n_subjects": len(per_subject)}
            metrics_dir = os.path.join(dose_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            write_per_subject_csv(
                os.path.join(metrics_dir, f"{arm}_per_subject.csv"),
                per_subject)
        results[str(dose)] = row

    print("\n" + "=" * 88)
    print(f"{'Dose':>5s} | {'BL PSNR':>10s} | {'TTA PSNR':>10s} | "
          f"{'ΔPSNR':>8s} | {'BL SSIM':>9s} | {'TTA SSIM':>9s} | "
          f"{'ΔSSIM':>8s}")
    print("-" * 88)
    for dose in sorted(results, key=lambda x: int(x)):
        row = results[dose]
        if "baseline" not in row or "tta" not in row:
            continue
        bl, tta = row["baseline"], row["tta"]
        print(f"{dose:>5s} | {bl['psnr']:10.4f} | {tta['psnr']:10.4f} | "
              f"{tta['psnr'] - bl['psnr']:+8.4f} | {bl['ssim']:9.5f} | "
              f"{tta['ssim']:9.5f} | {tta['ssim'] - bl['ssim']:+8.5f}")
    print("=" * 88)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result_dir", default=None,
                        help="A single directory with *_result.img / *_hr.img")
    parser.add_argument("--root", default=None,
                        help="Output root containing dose_*/{baseline,tta}")
    parser.add_argument("--doses", default=None,
                        help="Comma-separated doses for --root (default: all)")
    parser.add_argument("--arms", default="baseline,tta")
    parser.add_argument("--pred_suffix", default="_result.img")
    parser.add_argument("--gt_suffix", default="_hr.img")
    parser.add_argument("--background_threshold", type=float,
                        default=BACKGROUND_THRESHOLD)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    if not args.result_dir and not args.root:
        parser.error("provide --result_dir or --root")

    kwargs = dict(pred_suffix=args.pred_suffix, gt_suffix=args.gt_suffix,
                  background_threshold=args.background_threshold)

    if args.result_dir:
        psnr, ssim, nmse, per_subject = evaluate_results_dir(
            args.result_dir, **kwargs)
        print(f"\nAverage PSNR: {psnr:.6f} SSIM: {ssim:.6f} NMSE: {nmse:.6f}")
        results = {"psnr": psnr, "ssim": ssim, "nmse": nmse,
                   "n_subjects": len(per_subject)}
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
    else:
        doses = [d.strip() for d in args.doses.split(",")] if args.doses else None
        arms = tuple(a.strip() for a in args.arms.split(","))
        results = evaluate_root(args.root, doses=doses, arms=arms, **kwargs)
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
