"""
train.py

Orchestrator for the RealTimeGuard ML training pipeline.

Pipeline:
    Load CSV
      └─► Sample (development mode)
            └─► DataPreprocessor.preprocess()
                  └─► Train / Test Split
                        └─► build_model()
                              └─► ModelTrainer.train() + save_artifacts()
                                    └─► ModelEvaluator.evaluate() + generate_reports()
                                          └─► InferenceBenchmark.run() + save()
                                                └─► Save model_metadata.json

Usage:
    python -m src.train
"""

import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.benchmark import InferenceBenchmark
from src.config import (
    DATA_PATH,
    DEVELOPMENT_MODE,
    MODEL_DIR,
    MODEL_PARAMS,
    MODEL_VERSION,
    NORMAL_SAMPLE_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluator import ModelEvaluator
from src.model import build_model
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer
from src.utils import get_feature_list, save_json, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("RealTimeGuard — Training Pipeline  v%s", MODEL_VERSION)
    logger.info("=" * 60)

    # ──────────────────────────────────────────────
    # 1. Load dataset
    # ──────────────────────────────────────────────
    logger.info("Loading dataset from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # ──────────────────────────────────────────────
    # 2. Development-mode sampling
    # ──────────────────────────────────────────────
    if DEVELOPMENT_MODE:
        logger.info("Development mode: sampling dataset …")
        fraud_df = df[df["isFraud"] == 1]
        normal_df = df[df["isFraud"] == 0].sample(
            n=NORMAL_SAMPLE_SIZE, random_state=RANDOM_STATE
        )
        df = (
            pd.concat([normal_df, fraud_df])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        logger.info(
            "Sampled: %d normal | %d fraud",
            len(normal_df),
            len(fraud_df),
        )

    # ──────────────────────────────────────────────
    # 3. Preprocess
    # ──────────────────────────────────────────────
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df, training=True)
    logger.info("Preprocessing complete. Shape: %s", df.shape)

    # ──────────────────────────────────────────────
    # 4. Train / test split
    # ──────────────────────────────────────────────
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split: %d train | %d test (stratified)",
        len(X_train),
        len(X_test),
    )

    # ──────────────────────────────────────────────
    # 5. Build model
    # ──────────────────────────────────────────────
    model = build_model(scale_pos_weight=1.0)  # Trainer will override this

    # ──────────────────────────────────────────────
    # 6. Train
    # ──────────────────────────────────────────────
    trainer = ModelTrainer(model=model, preprocessor=preprocessor)
    training_info = trainer.train(X_train, y_train)
    trainer.save_artifacts(X_train)

    # ──────────────────────────────────────────────
    # 7. Evaluate
    # ──────────────────────────────────────────────
    evaluator = ModelEvaluator(model=model, X_test=X_test, y_test=y_test)
    metrics = evaluator.evaluate()
    evaluator.generate_reports()

    # ──────────────────────────────────────────────
    # 8. Benchmark
    # ──────────────────────────────────────────────
    benchmark = InferenceBenchmark(model=model, X_sample=X_test)
    benchmark_results = benchmark.run()
    benchmark.save()

    # ──────────────────────────────────────────────
    # 9. Save model metadata
    # ──────────────────────────────────────────────
    feature_list = get_feature_list(X_train)

    model_metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "development_mode": DEVELOPMENT_MODE,
        "random_seed": RANDOM_STATE,
        "sample_sizes": {
            "normal": int((y_train == 0).sum() + (y_test == 0).sum()),
            "fraud": int((y_train == 1).sum() + (y_test == 1).sum()),
        },
        "split": {
            "train": len(X_train),
            "test": len(X_test),
            "test_size": TEST_SIZE,
        },
        "feature_list": feature_list,
        "n_features": len(feature_list),
        "model_parameters": MODEL_PARAMS,
        "training_info": training_info,
        "metrics": metrics,
        "benchmark": benchmark_results,
        "preprocessing_version": "1.0",
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    save_json(model_metadata, metadata_path)
    logger.info("Model metadata saved → %s", metadata_path)

    # ──────────────────────────────────────────────
    # 10. Save training_metrics.json (backwards compat)
    # ──────────────────────────────────────────────
    compat_metrics = {k: v for k, v in metrics.items()
                      if k != "confusion_matrix"}
    save_json(compat_metrics, MODEL_DIR / "training_metrics.json")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(
        "  Recall    : %.4f  |  ROC-AUC : %.4f",
        metrics["recall"],
        metrics["roc_auc"],
    )
    logger.info(
        "  p99 latency : %.4f ms  |  Throughput : %.2f pred/sec",
        benchmark_results["p99_ms"],
        benchmark_results["throughput_per_second"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()