import pandas as pd
from typing import Tuple, Optional

def load_and_prepare_data(file) -> Tuple[Optional[pd.DataFrame], Optional[list]]:
    """
    Load and prepare GA4 data from CSV file
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple containing:
        - DataFrame with processed data
        - List of unique channels
        Returns (None, None) if error occurs
    """
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate required columns - we'll make this more flexible
        required_columns = {'date', 'channel'}
        metric_columns = {'sessions', 'orders', 'revenue'}
        
        # Check for absolute minimum requirements
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            print(f"Missing required columns: {missing}")
            return None, None
        
        # Check for at least one metric column
        available_metrics = metric_columns & set(df.columns)
        if not available_metrics:
            print(f"Need at least one of these metric columns: {metric_columns}")
            return None, None
        
        # Parse the 'date' column with dayfirst=True for DD/MM/YYYY format
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        
        # Check if any dates failed to parse
        if df['date'].isnull().any():
            print("Date parsing failed. Ensure dates are in DD/MM/YYYY format")
            return None, None
        
        # Get unique channels
        channels = df['channel'].unique().tolist()
        
        return df, channels
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None