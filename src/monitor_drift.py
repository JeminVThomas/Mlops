from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    print("Loading Iris dataset...")

    iris = load_iris(as_frame=True)

    # Reference dataset
    reference = iris.frame.sample(
        n=100, random_state=42
    )

    # Simulated production dataset
    current = iris.frame.sample(
        n=100, random_state=99
    ).copy()

    # Artificially introduce drift
    current["sepal length (cm)"] *= 1.5

    print("Running Evidently drift detection...")

    report = Report(
        metrics=[DataDriftPreset()]
    )

    snapshot = report.run(
        reference_data=reference,
        current_data=current
    )

    # Generate HTML dashboard
    snapshot.save_html("drift_report.html")

    # Extract metric results
    results = snapshot.dict()

    dataset_drift_share = 0.0
    drift_detected = False

    for metric in results["metrics"]:

        if metric["metric_name"].startswith(
            "DriftedColumnsCount"
        ):

            dataset_drift_share = metric["value"]["share"]

            # Trigger if at least 20% of columns drift
            drift_detected = dataset_drift_share >= 0.20

            break

    print("\n========== DRIFT RESULTS ==========")

    print(f"Drift Share: {dataset_drift_share:.2%}")

    print(f"Dataset Drift Detected: {drift_detected}")

    print("===================================")

    return drift_detected


if __name__ == "__main__":
    check_data_drift()
