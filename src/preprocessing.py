import pandas as pd
import os

def preprocess_data(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Filter for English tickets only (as specified in project plan)
    if 'language' in df.columns:
        initial_len = len(df)
        df = df[df['language'] == 'en'].copy()
        print(f"Filtered for English tickets: {initial_len} -> {len(df)} rows")

    # 2. Drop rows with missing mandatory fields
    df = df.dropna(subset=['subject', 'body', 'queue', 'priority']).copy()
    print(f"Dropped rows with missing targets/text. Remaining: {len(df)} rows")

    # 3. Construct the combined text feature: Type + Tags + Subject + Body
    def combine_text(row):
        parts = []
        if pd.notna(row.get('type')):
            parts.append(f"[{row['type']}]")
        
        tags = [str(row[f'tag_{i}']) for i in range(1, 9) if f'tag_{i}' in df.columns and pd.notna(row[f'tag_{i}'])]
        if tags:
            parts.append(f"[Tags: {', '.join(tags)}]")
            
        parts.append(f"Subject: {row['subject']}")
        parts.append(f"Body: {row['body']}")
        
        return " | ".join(parts)

    df['text'] = df.apply(combine_text, axis=1)

    # 4. Select only the columns needed for ML training
    final_df = df[['text', 'queue', 'priority']]

    # 5. Export to processed folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved processed dataset to {output_path}")
    print("\nSample processed data:")
    print(final_df.head(2))

if __name__ == "__main__":
    raw_path = "data/raw/customer_support_tickets.csv"
    processed_path = "data/processed/tickets.csv"
    preprocess_data(raw_path, processed_path)