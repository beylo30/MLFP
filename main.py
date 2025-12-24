from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data import load_cifar10, make_tf_datasets
from src.model import build_resnet_small, compile_with_cosine_sgd, build_data_augmentation
from src.train import train as train_loop
from src.eval import evaluate, plot_training_curves, classification_report_and_cm
from src.svm_baseline import extract_cnn_features, svm_on_features
from src.viz import compare_augmentation, pca_3d_plot
from src.predict import predict_image_from_url, show_image_from_url
from src.utils import Paths, set_reproducibility, gpu_available

DEFAULT_EXTERNAL_URL = (
    "https://www.lamborghini.com/sites/it-en/files/DAM/lamborghini/"
    "facelift_2019/homepage/families-gallery/2023/revuelto/revuelto_m.png"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-10: ResNet-style CNN + SVM baseline + PCA visualization "
            "(projectized from your notebook)."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Run everything in the same order as the notebook
    p_all = sub.add_parser("all", help="Run the full notebook pipeline in order.")
    p_all.add_argument("--epochs", type=int, default=10)
    p_all.add_argument("--batch-size", type=int, default=128)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--pca-n", type=int, default=6000)
    p_all.add_argument("--aug-index", type=int, default=56)
    p_all.add_argument("--url", type=str, default=DEFAULT_EXTERNAL_URL)
    p_all.add_argument(
        "--force-train",
        action="store_true",
        help="Train even if a saved model already exists.",
    )

    # Train only
    p_train = sub.add_parser("train", help="Train the CNN and save the best model.")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--seed", type=int, default=42)

    # Eval only
    p_eval = sub.add_parser("eval", help="Evaluate the best saved model on the test set.")
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--model-path", type=str, default="outputs/models/best_cifar10_model.keras")

    # Report only
    p_rep = sub.add_parser("report", help="Print classification report + plot confusion matrix.")
    p_rep.add_argument("--batch-size", type=int, default=128)
    p_rep.add_argument("--model-path", type=str, default="outputs/models/best_cifar10_model.keras")

    # SVM only
    p_svm = sub.add_parser("svm", help="Train an SVM on CNN features (baseline).")
    p_svm.add_argument("--epochs", type=int, default=10, help="If model doesn't exist, train for this many epochs.")
    p_svm.add_argument("--batch-size", type=int, default=128)
    p_svm.add_argument("--model-path", type=str, default="outputs/models/best_cifar10_model.keras")

    # PCA only
    p_pca = sub.add_parser("pca", help="3D PCA visualization of CNN features.")
    p_pca.add_argument("--n", type=int, default=6000, help="Number of points to plot.")
    p_pca.add_argument("--batch-size", type=int, default=128)
    p_pca.add_argument("--model-path", type=str, default="outputs/models/best_cifar10_model.keras")

    # Augmentation compare only
    p_aug = sub.add_parser("augment", help="Show original vs augmented image (from training set).")
    p_aug.add_argument("--index", type=int, default=56)

    # Predict external image only
    p_pred = sub.add_parser("predict", help="Predict a class for an external image URL.")
    p_pred.add_argument("--url", type=str, default=DEFAULT_EXTERNAL_URL)
    p_pred.add_argument("--model-path", type=str, default="outputs/models/best_cifar10_model.keras")

    return parser


def _resolve_model_path(paths: Paths, model_path_str: str) -> Path:
    p = Path(model_path_str)
    if not p.is_absolute():
        p = paths.project_root / p
    return p


def _load_or_train(
    paths: Paths,
    data,
    epochs: int,
    batch_size: int,
    seed: int,
    force_train: bool = False,
) -> keras.Model:
    model_path = paths.models_dir / "best_cifar10_model.keras"
    if model_path.exists() and not force_train:
        print(f"Loading saved model: {model_path}")
        return keras.models.load_model(model_path)

    print("No saved model found (or --force-train set). Training a new model...")
    set_reproducibility(seed)

    train_ds, val_ds, _ = make_tf_datasets(
        data.X_train, data.y_train,
        data.X_val, data.y_val,
        data.X_test, data.y_test,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_resnet_small(num_classes=len(data.classes))
    compile_with_cosine_sgd(model, train_ds=train_ds, epochs=epochs)

    history = train_loop(model, train_ds, val_ds, epochs=epochs, checkpoint_path=model_path)
    plot_training_curves(history, save_path=paths.figures_dir / "training_curves.png")

    return keras.models.load_model(model_path)


def run_all_pipeline(paths: Paths, args: argparse.Namespace) -> None:
    """
    Runs the same sections in the same order as the notebook:
    1) Load data + build pipelines
    2) Train CNN
    3) Evaluate on official test set
    4) Classification report + confusion matrix
    5) SVM baseline on CNN features
    6) PCA 3D visualization (from SVM train features)
    7) Compare original vs augmented image
    8) Predict external image
    """
    print("TensorFlow:", tf.__version__)
    print("GPU available:", gpu_available())

    data = load_cifar10(seed=args.seed)

    # 2) Train / load
    model = _load_or_train(
        paths=paths,
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        force_train=args.force_train,
    )

    # Datasets (used for eval/report/SVM)
    train_ds, val_ds, test_ds = make_tf_datasets(
        data.X_train, data.y_train,
        data.X_val, data.y_val,
        data.X_test, data.y_test,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 3) Evaluate
    test_loss, test_acc = evaluate(model, test_ds)
    print(f"\nOfficial TEST accuracy: {test_acc:.4f} (loss={test_loss:.4f})")

    # 4) Report + confusion matrix
    classification_report_and_cm(
        model=model,
        test_ds=test_ds,
        y_test=data.y_test,
        class_names=data.classes,
        save_path=paths.figures_dir / "confusion_matrix.png",
    )
    print(f"Saved confusion matrix to: {paths.figures_dir / 'confusion_matrix.png'}")

    # 5) SVM baseline (extract features once)
    autotune = tf.data.AUTOTUNE
    train_ds_noshuf = (
        tf.data.Dataset.from_tensor_slices((data.X_train, data.y_train))
        .batch(args.batch_size)
        .prefetch(autotune)
    )

    train_f, val_f, test_f = extract_cnn_features(model, train_ds_noshuf, val_ds, test_ds)
    svm_res = svm_on_features(
        train_features=train_f,
        y_train=data.y_train,
        val_features=val_f,
        y_val=data.y_val,
        test_features=test_f,
        y_test=data.y_test,
    )
    print("\nSVM baseline results:")
    print(" - best params:", svm_res.best_params)
    print(f" - val accuracy:  {svm_res.val_accuracy:.4f}")
    print(f" - test accuracy: {svm_res.test_accuracy:.4f}")

    # 6) PCA 3D visualization (from the SVM train features, like the notebook)
    n = min(args.pca_n, train_f.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(train_f.shape[0], size=n, replace=False)
    feat_subset = train_f[idx]
    y_subset = data.y_train[idx]
    pca_3d_plot(feat_subset, y_subset, save_path=paths.figures_dir / "pca_3d.png")
    print(f"Saved PCA 3D plot to: {paths.figures_dir / 'pca_3d.png'}")

    # 7) Compare original vs augmented image
    aug = build_data_augmentation()
    img = data.X_train[args.aug_index]
    compare_augmentation(aug, img, save_path=paths.figures_dir / "augmentation_compare.png")
    print(f"Saved augmentation comparison to: {paths.figures_dir / 'augmentation_compare.png'}")

    # 8) Predict external image
    pred = predict_image_from_url(model, args.url, class_names=data.classes)
    print(f"\nExternal image prediction: {pred.top_class} ({pred.top_prob*100:.2f}%)")
    show_image_from_url(args.url, title=f"Pred: {pred.top_class} ({pred.top_prob*100:.1f}%)")


def main() -> None:
    parser = build_parser()

    # If user runs `python main.py` with no args, run the full pipeline.
    if len(sys.argv) == 1:
        args = parser.parse_args(["all"])
    else:
        args = parser.parse_args()

    paths = Paths.from_root(Path(__file__).resolve().parent)

    if args.cmd in (None, "all"):
        run_all_pipeline(paths, args)
        return

    # Shared: load data once
    data = load_cifar10(seed=42)

    if args.cmd == "train":
        set_reproducibility(args.seed)
        train_ds, val_ds, test_ds = make_tf_datasets(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            data.X_test, data.y_test,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        model = build_resnet_small(num_classes=len(data.classes))
        compile_with_cosine_sgd(model, train_ds=train_ds, epochs=args.epochs)

        model_path = paths.models_dir / "best_cifar10_model.keras"
        history = train_loop(model, train_ds, val_ds, epochs=args.epochs, checkpoint_path=model_path)

        best_model = keras.models.load_model(model_path)
        test_loss, test_acc = evaluate(best_model, test_ds)
        print(f"\nOfficial TEST accuracy: {test_acc:.4f} (loss={test_loss:.4f})")

        plot_training_curves(history, save_path=paths.figures_dir / "training_curves.png")
        return

    if args.cmd == "eval":
        model_path = _resolve_model_path(paths, args.model_path)
        _, _, test_ds = make_tf_datasets(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            data.X_test, data.y_test,
            batch_size=args.batch_size,
        )
        model = keras.models.load_model(model_path)
        test_loss, test_acc = evaluate(model, test_ds)
        print(f"\nOfficial TEST accuracy: {test_acc:.4f} (loss={test_loss:.4f})")
        return

    if args.cmd == "report":
        model_path = _resolve_model_path(paths, args.model_path)
        _, _, test_ds = make_tf_datasets(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            data.X_test, data.y_test,
            batch_size=args.batch_size,
        )
        model = keras.models.load_model(model_path)
        classification_report_and_cm(
            model=model,
            test_ds=test_ds,
            y_test=data.y_test,
            class_names=data.classes,
            save_path=paths.figures_dir / "confusion_matrix.png",
        )
        print(f"Saved confusion matrix to: {paths.figures_dir / 'confusion_matrix.png'}")
        return

    if args.cmd == "svm":
        model_path = _resolve_model_path(paths, args.model_path)
        if model_path.exists():
            model = keras.models.load_model(model_path)
        else:
            model = _load_or_train(paths, data, epochs=args.epochs, batch_size=args.batch_size, seed=42)

        autotune = tf.data.AUTOTUNE
        train_ds_noshuf = (
            tf.data.Dataset.from_tensor_slices((data.X_train, data.y_train))
            .batch(args.batch_size)
            .prefetch(autotune)
        )
        _, val_ds, test_ds = make_tf_datasets(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            data.X_test, data.y_test,
            batch_size=args.batch_size,
        )

        train_f, val_f, test_f = extract_cnn_features(model, train_ds_noshuf, val_ds, test_ds)
        res = svm_on_features(train_f, data.y_train, val_f, data.y_val, test_f, data.y_test)
        print("SVM best params:", res.best_params)
        print(f"Val acc:  {res.val_accuracy:.4f}")
        print(f"Test acc: {res.test_accuracy:.4f}")
        return

    if args.cmd == "pca":
        model_path = _resolve_model_path(paths, args.model_path)
        model = keras.models.load_model(model_path)

        autotune = tf.data.AUTOTUNE
        train_ds_noshuf = (
            tf.data.Dataset.from_tensor_slices((data.X_train, data.y_train))
            .batch(args.batch_size)
            .prefetch(autotune)
        )
        _, val_ds, test_ds = make_tf_datasets(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
            data.X_test, data.y_test,
            batch_size=args.batch_size,
        )

        train_f, _, _ = extract_cnn_features(model, train_ds_noshuf, val_ds, test_ds)
        n = min(args.n, train_f.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(train_f.shape[0], size=n, replace=False)
        pca_3d_plot(train_f[idx], data.y_train[idx], save_path=paths.figures_dir / "pca_3d.png")
        print(f"Saved PCA 3D plot to: {paths.figures_dir / 'pca_3d.png'}")
        return

    if args.cmd == "augment":
        aug = build_data_augmentation()
        img = data.X_train[args.index]
        compare_augmentation(aug, img, save_path=paths.figures_dir / "augmentation_compare.png")
        print(f"Saved augmentation comparison to: {paths.figures_dir / 'augmentation_compare.png'}")
        return

    if args.cmd == "predict":
        model_path = _resolve_model_path(paths, args.model_path)
        model = keras.models.load_model(model_path)
        pred = predict_image_from_url(model, args.url, class_names=data.classes)
        print(f"\nExternal image prediction: {pred.top_class} ({pred.top_prob*100:.2f}%)")
        show_image_from_url(args.url, title=f"Pred: {pred.top_class} ({pred.top_prob*100:.1f}%)")
        return

    raise SystemExit(f"Unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()
