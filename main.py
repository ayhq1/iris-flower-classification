from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "iris.xlsx"


class DataHandler:
    """Load and explore the Iris dataset."""

    COLUMNS = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    def __init__(self, filepath=DATASET_PATH):
        self.filepath = Path(filepath)
        self.data = None

    def load_data(self):
        try:
            # The source Excel file has no header row.
            self.data = pd.read_excel(
                self.filepath,
                header=None,
                names=self.COLUMNS,
            )
            print(f"Dataset successfully loaded: {len(self.data)} rows.")
            return self.data
        except FileNotFoundError:
            print(f"Error: Could not find '{self.filepath}'.")
            return None
        except Exception as exc:
            print(f"An error occurred while loading the dataset: {exc}")
            return None

    def explore_data(self):
        if self.data is None:
            return

        print("\n--- Data Exploration ---")
        print(self.data.head())
        print(self.data.describe())

        plt.figure(figsize=(8, 6))
        colors = {
            "Iris-setosa": "red",
            "Iris-versicolor": "green",
            "Iris-virginica": "blue",
        }

        for species, color in colors.items():
            subset = self.data[self.data["species"] == species]
            plt.scatter(
                subset["sepal_length"],
                subset["sepal_width"],
                c=color,
                label=species,
                alpha=0.7,
            )

        plt.title("Iris Dataset: Sepal Length vs Sepal Width")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Sepal Width (cm)")
        plt.legend()
        plt.tight_layout()
        plt.show()


class IrisModel:
    """Train, evaluate, and use a Logistic Regression classifier."""

    FEATURE_COLUMNS = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression(max_iter=200)
        self.X_test = None
        self.y_test = None

    def prepare_and_train(self):
        X = self.data[self.FEATURE_COLUMNS]
        y = self.data["species"]

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.model.fit(X_train, y_train)
        print("\nModel training complete.")

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%\n")
        return accuracy

    def predict_species(self, features):
        sample = pd.DataFrame([features], columns=self.FEATURE_COLUMNS)
        return self.model.predict(sample)[0]


class IrisApp:
    """Simple Tkinter interface for classifying an Iris flower."""

    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Iris Flower Classifier")
        self.root.geometry("330x270")

        labels = [
            "Sepal Length (cm):",
            "Sepal Width (cm):",
            "Petal Length (cm):",
            "Petal Width (cm):",
        ]

        self.entries = []
        for row, label_text in enumerate(labels):
            tk.Label(self.root, text=label_text).grid(
                row=row, column=0, padx=10, pady=10, sticky="e"
            )
            entry = tk.Entry(self.root)
            entry.grid(row=row, column=1, padx=10)
            self.entries.append(entry)

        tk.Button(
            self.root,
            text="Predict Species",
            command=self.handle_prediction,
        ).grid(row=4, column=0, columnspan=2, pady=15)

    def handle_prediction(self):
        try:
            features = [float(entry.get()) for entry in self.entries]
            predicted_species = self.model.predict_species(features)
            messagebox.showinfo(
                "Prediction Result",
                f"The predicted species is:\n\n{predicted_species}",
            )
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Please enter valid numbers in all fields.",
            )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    data_handler = DataHandler()
    iris_df = data_handler.load_data()

    if iris_df is not None:
        data_handler.explore_data()

        classifier = IrisModel(iris_df)
        classifier.prepare_and_train()
        classifier.evaluate_model()

        print("Launching the GUI application...")
        IrisApp(classifier).run()
