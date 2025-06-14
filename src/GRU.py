#GRU on Fake and Real News Dataset

#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Helper function to read and combine Fake and Real CSVs
def load_fake_news_dataset():
    fake = pd.read_csv("/fake-and-real-news-dataset/Fake.csv")
    true = pd.read_csv("/fake-and-real-news-dataset/True.csv")
    fake['label'] = 1
    true['label'] = 0
    df = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)
    return df

# Tokenization and preprocessing
def preprocess_data(texts, labels, max_words=10000, max_len=200, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded, labels, tokenizer

# GRU model builder
def build_gru_model(output_units, output_activation, input_length=200):
    model = Sequential([
        Embedding(10000, 100, input_length=input_length),
        GRU(64),
        Dropout(0.5),
        Dense(output_units, activation=output_activation)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if output_units == 1 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Evaluation function
def evaluate_model(model, X_test, y_test, is_multiclass=False, label_names=None, title="Confusion Matrix"):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if is_multiclass else (y_pred > 0.5).astype("int32").flatten()
	 acc = accuracy_score(y_test, y_pred_classes)
    print(f"\nAccuracy: {acc*100:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_classes, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

df_fake = load_fake_news_dataset()
X_fake, y_fake = df_fake['text'], df_fake['label']
X_fake_pad, y_fake, tokenizer = preprocess_data(X_fake, y_fake)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fake_pad, y_fake, test_size=0.2, random_state=42)

model_fake = build_gru_model(output_units=1, output_activation='sigmoid')
model_fake.fit(Xf_train, yf_train, epochs=5, batch_size=32, validation_data=(Xf_test, yf_test))
evaluate_model(model_fake, Xf_test, yf_test, is_multiclass=False, label_names=["REAL", "FAKE"], title="Fake News Confusion Matrix")

#GRU on LIAR Dataset

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load and preprocess the LIAR dataset more thoroughly
def load_and_preprocess_liar():
    # Load dataset
    train_df = pd.read_csv('/liar-dataset/train.tsv', sep='\t', header=None)
    test_df = pd.read_csv('/liar-dataset/test.tsv', sep='\t', header=None)
    valid_df = pd.read_csv('/liar-dataset/valid.tsv', sep='\t', header=None)
    
    # Combine all splits
    df = pd.concat([train_df, test_df, valid_df])
    
    # Assign column names
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
               'state_info', 'party_affiliation', 'barely_true', 'false', 
               'half_true', 'mostly_true', 'pants_on_fire', 'context']
    df.columns = columns
    
    # Focus on statement text and label
    df = df[['statement', 'label', 'subject', 'speaker', 'party_affiliation']]
    
    # Clean text more thoroughly
    df['statement'] = df['statement'].str.lower()
    df['statement'] = df['statement'].str.replace('[^\w\s]', '')
    df['statement'] = df['statement'].str.replace('\d+', '')
    
    # Handle missing values
    df.fillna('unknown', inplace=True)
    
    # Combine text with metadata
    df['combined_text'] = df['statement'] + ' ' + df['subject'] + ' ' + df['speaker'] + ' ' + df['party_affiliation']
    
    return df

# Enhanced preprocessing
def enhanced_preprocessing(df, max_words=20000, max_len=200):
    # Balance classes by undersampling
    class_counts = Counter(df['label'])
     min_count = min(class_counts.values())
    balanced_df = pd.concat([
        df[df['label'] == cls].sample(min_count, random_state=42)
        for cls in class_counts.keys()
    ])
    
    # Encode labels
    label_encoder = LabelEncoder()
    balanced_df['label_encoded'] = label_encoder.fit_transform(balanced_df['label'])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['combined_text'], balanced_df['label_encoded'], 
        test_size=0.2, random_state=42, stratify=balanced_df['label_encoded'])
    
    # Tokenize text with more sophisticated preprocessing
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder

# Enhanced GRU model with more sophisticated architecture
def build_enhanced_gru_model(vocab_size, embedding_dim=256, gru_units=128, max_len=200):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                 input_length=max_len, mask_zero=True),
        SpatialDropout1D(0.3),
        Bidirectional(GRU(gru_units, return_sequences=True, recurrent_dropout=0.2)),
        Dropout(0.4),
        Bidirectional(GRU(gru_units//2, recurrent_dropout=0.2)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(6, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
def enhanced_evaluation(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nEnhanced Classification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_, digits=4))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_encoder.classes_, 
               yticklabels=label_encoder.classes_)
    plt.title('Enhanced Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nEnhanced Model Accuracy: {accuracy*100:.4f}")
    
    return accuracy

# Plot training history with more details
def enhanced_plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Enhanced Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Enhanced Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main execution with enhancements
def main():
    # Load and preprocess data with enhancements
    df = load_and_preprocess_liar()
    X_train, X_test, y_train, y_test, tokenizer, label_encoder = enhanced_preprocessing(df)
     vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 256
    gru_units = 128
    max_len = 200
    batch_size = 128
    epochs = 30
    
    # Build enhanced model
    model = build_enhanced_gru_model(vocab_size, embedding_dim, gru_units, max_len)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    enhanced_plot_history(history)
    accuracy = enhanced_evaluation(model, X_test, y_test, label_encoder)
    
    # Save the model for later use
    model.save('liar_gru_model.h5')
    print(f"Model saved with accuracy: {accuracy*100:.4f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure TensorFlow for better performance
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    main()