import os
import shutil
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

def main():
    print("Starting training process...")
    
    # 1. Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 2. Copy the dataset if it's not present locally
    local_csv_path = 'data/fake_job_postings.csv'
    source_csv_path = r'C:\Users\ANAMIKA\Downloads\fake_job_postings.csv\fake_job_postings.csv'
    
    if not os.path.exists(local_csv_path):
        if os.path.exists(source_csv_path):
            print(f"Copying dataset from {source_csv_path} to {local_csv_path}...")
            shutil.copy(source_csv_path, local_csv_path)
            print("Dataset copied successfully.")
        else:
            raise FileNotFoundError(f"Source dataset not found at {source_csv_path}")
    else:
        print("Local dataset already exists. Skipping copy.")

    # 3. Load dataset
    print("Loading dataset...")
    df = pd.read_csv(local_csv_path)
    
    # 4. Data Preprocessing (replicating notebook logic)
    print("Preprocessing data...")
    # Fill location NaNs with 'blank'
    df['location'] = df['location'].fillna('blank')
    
    # Filter US listings
    df_us = df[df['location'].str.contains("US")].copy()
    
    # Split location into state and city
    loc_splits = []
    for loc in df_us['location']:
        parts = loc.split(',')
        # Emulate pandas DataFrame conversion behaviour when list lengths vary
        # Fill missing dimensions with None
        state = parts[1] if len(parts) > 1 else None
        city = parts[2] if len(parts) > 2 else None
        loc_splits.append({'state': state, 'city': city})
        
    loc_df = pd.DataFrame(loc_splits)
    df_us = df_us.reset_index(drop=True)
    df_us = df_us.join(loc_df)
    
    # Drop rows with null city or state
    df_us = df_us[df_us['city'].notna()]
    df_us = df_us[df_us['state'].notna()]
    
    # Strip whitespace
    df_us['city'] = df_us['city'].str.strip()
    df_us['state'] = df_us['state'].str.strip()
    df_us['state_city'] = df_us['state'] + ", " + df_us['city']
    
    # 5. Compute Location Ratio (fraud / real)
    print("Computing location ratios...")
    # Group by state_city for fraudulent and non-fraudulent
    fraud_counts = df_us[df_us['fraudulent'] == 1].groupby('state_city').size()
    real_counts = df_us[df_us['fraudulent'] == 0].groupby('state_city').size()
    
    # Calculate ratio, matching division behaviour of notebook
    ratio_series = round(fraud_counts / real_counts, 2)
    
    # Create ratio mapping dataframe
    location_ratio_df = pd.DataFrame({'state_city': ratio_series.index, 'ratio': ratio_series.values})
    
    # Merge ratio back and fill NaNs with 0
    df_us = df_us.merge(location_ratio_df, on='state_city', how='left')
    df_us['ratio'] = df_us['ratio'].fillna(0)
    
    # Replace infinite values with ratio (if any division by zero occurred and not filled)
    df_us['ratio'] = df_us['ratio'].replace([np.inf, -np.inf], 0)
    
    # Export location ratio map as JSON
    ratio_dict = location_ratio_df.set_index('state_city')['ratio'].fillna(0).to_dict()
    # Handle infinite values in dictionary if any
    for k, v in ratio_dict.items():
        if not np.isfinite(v):
            ratio_dict[k] = 0.0
            
    with open('models/location_ratio.json', 'w') as f:
        json.dump(ratio_dict, f, indent=4)
    print("Location ratios saved to models/location_ratio.json")
    
    # 6. Concat Text Fields
    print("Concatenating text fields...")
    # Select columns to drop/keep matching notebook logic
    # The columns to concatenate are: title, location, company_profile, description, requirements, benefits, required_experience, required_education, industry, function
    text_cols = ['title', 'location', 'company_profile', 'description', 'requirements', 
                 'benefits', 'required_experience', 'required_education', 'industry', 'function']
    
    # Replicate .fillna(" ") from notebook
    df_us[text_cols] = df_us[text_cols].fillna(" ")
    
    # Concatenate the text columns with a space separator
    df_us['text'] = df_us[text_cols].agg(' '.join, axis=1)
    
    # Compute character count
    df_us['character_count'] = df_us['text'].apply(len)
    
    # Save clean dataset (optional)
    clean_csv_path = 'data/fake_job_postings_cleaned.csv'
    df_us.to_csv(clean_csv_path, index=False)
    print(f"Cleaned dataset saved to {clean_csv_path}")
    
    # 7. Model Training
    # The notebook trains on count_vectorizer with SGDClassifier(loss='log_loss')
    # and a numerical SGDClassifier on [telecommuting, ratio, character_count]
    print("Training models...")
    X_text = df_us['text'].values
    X_num = df_us[['telecommuting', 'ratio', 'character_count']].values
    y = df_us['fraudulent'].values
    
    # Fit vectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_text_transformed = vectorizer.fit_transform(X_text)
    
    # Fit Text Classifier
    print("Fitting text classifier (SGDClassifier)...")
    clf_log = SGDClassifier(loss='log_loss', random_state=53)
    clf_log.fit(X_text_transformed, y)
    
    # Fit Numerical Classifier
    print("Fitting numerical classifier (SGDClassifier)...")
    clf_num = SGDClassifier(loss='log_loss', random_state=53)
    clf_num.fit(X_num, y)
    
    # 8. Serialize Models
    print("Saving models...")
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/clf_log.pkl', 'wb') as f:
        pickle.dump(clf_log, f)
    with open('models/clf_num.pkl', 'wb') as f:
        pickle.dump(clf_num, f)
        
    print("Models saved successfully to models/ directory.")
    print("Training process completed.")

if __name__ == "__main__":
    main()
