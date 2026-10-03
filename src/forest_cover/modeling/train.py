"""Train a first Forest CoverType classification baseline."""

from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    features, labels = fetch_covtype(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        train_size=5_000,
        test_size=10_000,
        random_state=42,
        stratify=labels,
    )

    model = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(max_iter=1_000),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
