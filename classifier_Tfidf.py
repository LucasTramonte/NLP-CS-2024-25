import os
import logging
import pandas as pd
import argparse
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LanguageClassifier_TFIDF:
    def __init__(self, max_features=10000, test_size=0.2, random_state=42):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        logger.info("Initialization complete.")

    def preprocess_data(self, data: pd.DataFrame):
        """
        Encodes labels for training.
        """
        logger.info("Preprocessing data...")
        X = data['Text'].tolist()
        y = data['Label'].tolist()
        y = self.label_encoder.fit_transform(y)
        logger.info("Data preprocessing complete.")
        return X, y

    def train(self, X, y):
        """
        Trains the model using the provided data.
        """
        logger.info("Starting training...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.pipeline.fit(X_train, y_train)
        val_score = self.pipeline.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")
        self.save_model()
        logger.info("Training complete.")

    def save_model(self):
        os.makedirs('Assets/Outputs', exist_ok=True)
        joblib.dump(self.pipeline, 'Assets/Outputs/tfidf_model.pkl')
        joblib.dump(self.label_encoder, 'Assets/Outputs/label_encoder.pkl')
        self.is_trained = True
        logger.info("Model and LabelEncoder trained and saved successfully.")

    def load_model(self):
        model_path = 'Assets/Outputs/tfidf_model.pkl'
        label_encoder_path = 'Assets/Outputs/label_encoder.pkl'

        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            logger.info("Loading saved model and label encoder...")
            self.pipeline = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.is_trained = True
            logger.info("Model and label encoder loaded successfully.")
        else:
            logger.error("Model or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Model or label encoder file not found.")
        
    def evaluate(self, X, y):
        """
        Evaluates the model using a test dataset.
        """
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            self.load_model()

        logger.info("Starting evaluation...")
        predictions = self.pipeline.predict(X)
        report = classification_report(y, predictions, target_names=self.label_encoder.classes_)
        matrix = confusion_matrix(y, predictions)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))
        
        try:
            with open('Assets/Outputs/evaluation_report_TFIDF.txt', 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write(str(matrix))
            logger.info("Evaluation report saved to Assets/Outputs/evaluation_report_TFIDF.txt")
        except Exception as e:
            logger.error(f"Error while saving the evaluation report: {e}")
    
    def predict_language(self, text):
        """
        Predicts the language of a single input text.
        """
        if not self.is_trained:
            try:
                logger.info("Loading model and LabelEncoder for prediction...")
                self.pipeline = joblib.load('Assets/Outputs/tfidf_model.pkl')
                self.label_encoder = joblib.load('Assets/Outputs/label_encoder.pkl')
                self.is_trained = True
                logger.info("Model and LabelEncoder loaded successfully from 'Assets/Outputs/tfidf_model.pkl'.")
            except Exception as e:
                logger.error(f"Error loading the trained model or LabelEncoder: {e}")
                return None

        prediction = self.pipeline.predict([text])[0]
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_label

    def generate_submission(self, test_dataset_path, submission_path):
        """
        Generates a submission file for the Kaggle competition.
        """
        logger.info(f"Loading test dataset from {test_dataset_path}...")
        test_data = pd.read_csv(test_dataset_path)
        
        # Generate 'Id' column if it doesn't exist
        if 'ID' not in test_data.columns:
            test_data['ID'] = range(1, len(test_data) + 1)
        
        predictions = []

        logger.info("Starting predictions for the test dataset...")
        for text in tqdm(test_data['Text'], desc="Predicting languages"):
            predicted_language = self.predict_language(text)
            predictions.append(predicted_language)

        submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predictions})
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to {submission_path}")

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the dataset.
        """
        logger.info("Removing duplicate rows from the dataset...")
        initial_row_count = data.shape[0]
        data = data.drop_duplicates()
        final_row_count = data.shape[0]
        logger.info(f"Removed {initial_row_count - final_row_count} duplicate rows.")
        return data

def main():
    parser = argparse.ArgumentParser(description='Language Classification with TF-IDF')
    parser.add_argument('--train_dataset', type=str, help='Path to the CSV training dataset')
    parser.add_argument('--test_dataset', type=str, help='Path to the CSV test dataset')
    parser.add_argument('--predict', type=str, help='Text input to predict the language')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model')
    parser.add_argument('--submission', type=str, help='Path to save the submission file')
    args = parser.parse_args()

    classifier = LanguageClassifier_TFIDF()

    if args.train:
        if not args.train_dataset:
            raise ValueError("Please provide a training dataset path with --train_dataset when training the model.")
        
        logger.info(f"Loading training dataset from {args.train_dataset}...")
        data = pd.read_csv(args.train_dataset)
        data = classifier.remove_duplicates(data)
        X, y = classifier.preprocess_data(data)
        classifier.train(X, y)
        logger.info("Model trained successfully.")

    if args.predict:
        logger.info(f"Predicting language for input text: {args.predict}")
        predicted_language = classifier.predict_language(args.predict)
        logger.info(f"Predicted Language: {predicted_language}")

    # TO MODIFY : SPLIT INTO TRAIN AND TEST
    if args.evaluate:
        if not args.test_dataset:
            raise ValueError("Please provide a test dataset path with --test_dataset when evaluating the model.")
        
        logger.info(f"Loading test dataset from {args.test_dataset}...")
        data = pd.read_csv(args.test_dataset)
        data = classifier.remove_duplicates(data)
        X, y = classifier.preprocess_data(data)
        classifier.evaluate(X, y)
        logger.info("Evaluation completed.")
        print("Evaluation completed.")

    if args.submission:
        if not args.test_dataset:
            raise ValueError("Please provide a test dataset path with --test_dataset when generating the submission file.")
        
        logger.info("Generating submission file for the test dataset...")
        classifier.generate_submission(args.test_dataset, args.submission)
        logger.info("Submission file generated successfully.")

if __name__ == '__main__':
    main()