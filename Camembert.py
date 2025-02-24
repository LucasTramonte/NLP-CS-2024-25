import os
import zipfile
import logging
import pandas as pd
import argparse
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LanguageDataset(Dataset):
    """Custom Dataset for handling tokenized inputs and labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


class LanguageClassifierBERT:
    """Language Classifier using CamemBERT for sequence classification."""

    def __init__(self, model_name='camembert-base', max_len=128, batch_size=128, epochs=15):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_weights = None
        self.model = None  # Model will be initialized during preprocessing
        logger.info("Initialization complete.")

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """Tokenizes input text data and encodes labels for training."""
        logger.info("Preprocessing data...")
        X = data['Text'].tolist()
        y = data['Label'].tolist()
        y = self.label_encoder.fit_transform(y)

        # Initialize the model with the correct number of labels
        num_labels = len(self.label_encoder.classes_)
        self.model = CamembertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(self.device)

        # Tokenize input text
        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')

        # Compute class weights for imbalanced datasets
        class_counts = torch.bincount(torch.tensor(y))
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()  # Normalize
        self.class_weights = class_weights.to(self.device)

        # Validate tokenized inputs and labels
        if len(encodings['input_ids']) != len(y):
            raise ValueError("Tokenized inputs and labels have mismatched lengths.")

        logger.info("Data preprocessing complete.")
        return encodings.to(self.device), torch.tensor(y).to(self.device)

    def train(self, encodings, y):
        """Trains the BERT model using the provided data with mixed precision training."""
        logger.info("Starting training...")
        dataset = LanguageDataset(encodings, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        best_loss = float('inf')

        # Initialize GradScaler for mixed precision training
        scaler = GradScaler('cuda')

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                # Mixed precision training
                with autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = criterion(outputs.logits, batch['labels'])

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")

            # Save the best model
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self.save_model()

        logger.info("Training complete.")

    def save_model(self):
        """Saves the model, tokenizer, and label encoder to disk."""
        model_dir = 'Assets/Outputs/Models/Camembert'
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(os.path.join(model_dir, 'camembert_model'))
        self.tokenizer.save_pretrained(os.path.join(model_dir, 'camembert_model'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        self.is_trained = True
        logger.info("Model and LabelEncoder saved successfully.")

    def load_model(self):
        """Loads the model, tokenizer, and label encoder from disk."""
        model_dir = 'Assets/Outputs/Models/Camembert/camembert_model'
        label_encoder_path = 'Assets/Outputs/Models/Camembert/label_encoder.pkl'

        if not os.path.exists(model_dir) or not os.path.exists(label_encoder_path):
            logger.error("Model directory or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Model directory or label encoder file not found.")

        logger.info("Loading saved model, tokenizer, and label encoder...")
        self.model = CamembertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        self.label_encoder = joblib.load(label_encoder_path)
        self.is_trained = True
        logger.info("Model, tokenizer, and label encoder loaded successfully.")

    def predict_language(self, text: str) -> str:
        """Predicts the language of a single input text."""
        if not self.is_trained:
            self.load_model()

        encodings = self.tokenizer([text], truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        return predicted_label

    def generate_submission(self, test_dataset_path: str, submission_path: str):
        """Generates a submission file for the Kaggle competition."""
        logger.info(f"Loading test dataset from {test_dataset_path}...")
        test_data = pd.read_csv(test_dataset_path)

        # Generate 'Id' column if it doesn't exist
        if 'ID' not in test_data.columns:
            test_data['ID'] = range(1, len(test_data) + 1)

        logger.info("Starting predictions for the test dataset...")
        predictions = [self.predict_language(text) for text in tqdm(test_data['Text'], desc="Predicting languages")]

        submission_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predictions})
        submission_dir = 'Assets/Outputs/Submission'
        os.makedirs(submission_dir, exist_ok=True)
        submission_file_path = os.path.join(submission_dir, submission_path)
        submission_df.to_csv(submission_file_path, index=False)
        logger.info(f"Submission file saved to {submission_file_path}")

        # Create a zip file of the submission CSV
        zip_file_path = submission_file_path.replace('.csv', '.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(submission_file_path, os.path.basename(submission_file_path))
        logger.info(f"Submission file zipped to {zip_file_path}")

    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows from the dataset."""
        logger.info("Removing duplicate rows from the dataset...")
        initial_row_count = data.shape[0]
        data = data.drop_duplicates()
        final_row_count = data.shape[0]
        logger.info(f"Removed {initial_row_count - final_row_count} duplicate rows.")
        return data


def main():
    """Main function to handle command-line arguments and execute the pipeline."""
    parser = argparse.ArgumentParser(description='Language Classification with Camembert')
    parser.add_argument('--train_dataset', type=str, help='Path to the CSV training dataset')
    parser.add_argument('--test_dataset', type=str, help='Path to the CSV test dataset')
    parser.add_argument('--predict', type=str, help='Text input to predict the language')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--submission', type=str, help='Path to save the submission file')
    args = parser.parse_args()

    classifier = LanguageClassifierBERT()

    if args.train:
        if not args.train_dataset:
            raise ValueError("Please provide a training dataset path with --train_dataset when training the model.")
        logger.info(f"Loading training dataset from {args.train_dataset}...")
        data = pd.read_csv(args.train_dataset)
        data = classifier.remove_duplicates(data)
        encodings, y = classifier.preprocess_data(data)
        classifier.train(encodings, y)
        logger.info("Model trained successfully.")

    if args.predict:
        logger.info(f"Predicting language for input text: {args.predict}")
        predicted_language = classifier.predict_language(args.predict)
        logger.info(f"Predicted Language: {predicted_language}")

    if args.submission:
        if not args.test_dataset:
            raise ValueError("Please provide a test dataset path with --test_dataset when generating the submission file.")
        logger.info("Generating submission file for the test dataset...")
        classifier.generate_submission(args.test_dataset, args.submission)
        logger.info("Submission file generated successfully.")


if __name__ == '__main__':
    main()