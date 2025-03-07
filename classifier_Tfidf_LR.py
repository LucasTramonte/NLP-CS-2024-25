import os
import zipfile
import logging
import pandas as pd
import argparse
import joblib
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "Assets/Outputs/Models/LogisticRegression"
EVALUATION_DIR = "Assets/Outputs/Evaluation"
SUBMISSION_DIR = "Assets/Outputs/Submission"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

class LanguageClassifierTFIDF:
    """
    A language classifier using TF-IDF for feature extraction and Logistic Regression for classification.
    """

    def __init__(self, max_features: int = 10000, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize the classifier.

        Args:
            max_features (int): Maximum number of features for TF-IDF.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        logger.info("LanguageClassifierTFIDF initialized.")

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Preprocess the data by encoding labels.

        Args:
            data (pd.DataFrame): The dataset containing 'Text' and 'Label' columns.

        Returns:
            Tuple[List[str], List[int]]: A tuple of text data and encoded labels.
        """
        logger.info("Preprocessing data...")
        X = data['Text'].tolist()
        y = self.label_encoder.fit_transform(data['Label'].tolist())
        logger.info("Data preprocessing complete.")
        return X, y

    def train(self, X: List[str], y: List[int]) -> None:
        """
        Train the model using the provided data.

        Args:
            X (List[str]): List of text data.
            y (List[int]): List of encoded labels.
        """
        logger.info("Starting training...")
        with tqdm(total=1, desc="Training model") as pbar:
            self.pipeline.fit(X, y)
            pbar.update(1)
        self.save_model()
        logger.info("Training complete.")

    def save_model(self) -> None:
        """
        Save the trained model and label encoder to disk.
        """
        model_path = os.path.join(MODEL_DIR, 'tfidf_model.pkl')
        label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        self.is_trained = True
        logger.info(f"Model and LabelEncoder saved to {MODEL_DIR}.")

    def load_model(self) -> None:
        """
        Load the trained model and label encoder from disk.
        """
        model_path = os.path.join(MODEL_DIR, 'tfidf_model.pkl')
        label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

        if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
            logger.error("Model or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Model or label encoder file not found.")

        logger.info("Loading saved model and label encoder...")
        self.pipeline = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.is_trained = True
        logger.info("Model and label encoder loaded successfully.")

    def evaluate(self, X: List[str], y: List[int]) -> None:
        """
        Evaluate the model using a test dataset.

        Args:
            X (List[str]): List of text data.
            y (List[int]): List of encoded labels.
        """
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            self.load_model()

        logger.info("Starting evaluation...")
        predictions = [self.pipeline.predict([text])[0] for text in tqdm(X, desc="Evaluating model")]
        report = classification_report(y, predictions, target_names=self.label_encoder.classes_)
        matrix = confusion_matrix(y, predictions)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))

        evaluation_report_path = os.path.join(EVALUATION_DIR, 'evaluation_report_TFIDF.txt')
        try:
            with open(evaluation_report_path, 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write(str(matrix))
            logger.info(f"Evaluation report saved to {evaluation_report_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")

    def predict_language(self, text: str) -> Optional[str]:
        """
        Predict the language of a single input text.

        Args:
            text (str): The input text to classify.

        Returns:
            Optional[str]: The predicted language or None if an error occurs.
        """
        if not self.is_trained:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return None

        prediction = self.pipeline.predict([text])[0]
        return self.label_encoder.inverse_transform([prediction])[0]

    def generate_submission(self, test_dataset_path: str, submission_path: str) -> None:
        """
        Generate a submission file for the test dataset.

        Args:
            test_dataset_path (str): Path to the test dataset CSV file.
            submission_path (str): Name of the submission file to save.
        """
        logger.info(f"Loading test dataset from {test_dataset_path}...")
        test_data = pd.read_csv(test_dataset_path)

        # Generate 'ID' column if it doesn't exist
        if 'ID' not in test_data.columns:
            test_data['ID'] = range(1, len(test_data) + 1)

        logger.info("Starting predictions for the test dataset...")
        predictions = [self.predict_language(text) for text in tqdm(test_data['Text'], desc="Predicting languages")]

        submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predictions})
        submission_file_path = os.path.join(SUBMISSION_DIR, submission_path)
        submission_df.to_csv(submission_file_path, index=False)
        logger.info(f"Submission file saved to {submission_file_path}")

        # Create a zip file of the submission CSV
        zip_file_path = submission_file_path.replace('.csv', '.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(submission_file_path, os.path.basename(submission_file_path))
        logger.info(f"Submission file zipped to {zip_file_path}")

    @staticmethod
    def handle_missing_labels(data: pd.DataFrame) -> pd.DataFrame:
        """Removes rows with missing labels from the dataset."""
        logger.info("Removing rows with missing labels...")
        initial_row_count = data.shape[0]
        data = data.dropna(subset=['Label'])
        final_row_count = data.shape[0]
        logger.info(f"Removed {initial_row_count - final_row_count} rows with missing labels.")
        return data


def main():
    parser = argparse.ArgumentParser(description='Language Classification with TF-IDF')
    parser.add_argument('--train_dataset', type=str, help='Path to the CSV training dataset')
    parser.add_argument('--test_dataset', type=str, help='Path to the CSV test dataset')
    parser.add_argument('--predict', type=str, help='Text input to predict the language')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model')
    parser.add_argument('--submission', type=str, help='Name of the submission file to save')
    args = parser.parse_args()

    classifier = LanguageClassifierTFIDF()

    if args.train:
        if not args.train_dataset:
            raise ValueError("Please provide a training dataset path with --train_dataset when training the model.")
        data = pd.read_csv(args.train_dataset)
        data = classifier.handle_missing_labels(data)
        X, y = classifier.preprocess_data(data)
        classifier.train(X, y)

    if args.predict:
        predicted_language = classifier.predict_language(args.predict)
        logger.info(f"Predicted Language: {predicted_language}")

    if args.evaluate:
        if not args.train_dataset:
            raise ValueError("Please provide a train dataset path with --train_dataset when evaluating the model.")
        data = pd.read_csv(args.train_dataset)
        data = classifier.remove_duplicates(data)
        X, y = classifier.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=classifier.test_size, random_state=classifier.random_state)
        classifier.train(X_train, y_train)
        classifier.evaluate(X_test, y_test)

    if args.submission:
        if not args.test_dataset:
            raise ValueError("Please provide a test dataset path with --test_dataset when generating the submission file.")
        classifier.generate_submission(args.test_dataset, args.submission)


if __name__ == '__main__':
    main()