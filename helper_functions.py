# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy as sc 
from scipy.stats.mstats import winsorize 
import warnings 
warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use('seaborn-v0_8')
# Handling Missing values

def handle_missing_values(df, col_drop_threshold=0.60, row_drop_threshold=0.05, large_data_threshold=10000):
    """
    Evaluates columns with missing data and applies the optimal handling strategy.
    
    Parameters:
    - df: The input pandas DataFrame.
    - col_drop_threshold: Drop the entire column if missing values exceed this % (default 60%).
    - row_drop_threshold: Drop rows if missing values are below this % in large datasets (default 5%).
    - large_data_threshold: Minimum rows to consider the dataset "large" for row dropping (default 10,000).
    """
    cleaned_df = df.copy()
    decision_log = []
    
    # Identify columns that actually contain missing values
    cols_with_missing = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
    
    for col in cols_with_missing:
        missing_count = cleaned_df[col].isnull().sum()
        missing_pct = missing_count / len(cleaned_df)
        col_dtype = cleaned_df[col].dtype
        
        # Strategy 1: Column Removal (Too much missing data)
        # If a column is missing more than 60% of its data, imputing it introduces heavy bias.
        if missing_pct > col_drop_threshold:
            cleaned_df.drop(columns=[col], inplace=True)
            decision_log.append({
                'Column': col, 'DType': str(col_dtype), 'Missing %': round(missing_pct * 100, 2),
                'Strategy Applied': 'Column Removed (Exceeds Threshold)'
            })
            continue
            
        # Strategy 2: Row Removal (Negligible missing data in a large dataset)
        # If missing data is < 5% and the dataset is large, dropping rows preserves raw data integrity.
        if len(cleaned_df) >= large_data_threshold and missing_pct <= row_drop_threshold:
            cleaned_df.dropna(subset=[col], inplace=True)
            decision_log.append({
                'Column': col, 'DType': str(col_dtype), 'Missing %': round(missing_pct * 100, 2),
                'Strategy Applied': 'Rows Removed (Negligible % in Large Data)'
            })
            continue
            
        # Strategy 3: Dynamic Imputation for Numeric Data
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            # Check skewness to determine distribution shape
            skewness = cleaned_df[col].skew()
            
            if abs(skewness) > 1.5:
                # Highly skewed data (contains outliers or long tails): Use Median
                fill_value = cleaned_df[col].median()
                strategy = 'Median Imputation (Skewed Data / Outliers)'
            else:
                # Normally distributed data: Use Mean
                fill_value = cleaned_df[col].mean()
                strategy = 'Mean Imputation (Normal Distribution)'
                
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
            
        # Strategy 4: Imputation for Categorical/Text/Boolean Data
        else:
            mode_series = cleaned_df[col].mode()
            if not mode_series.empty:
                fill_value = mode_series.iloc[0]
                strategy = 'Mode Imputation (Categorical/Text)'
            else:
                fill_value = 'Unknown'
                strategy = "Filled with 'Unknown' (No Clear Mode)"
                
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
            
        decision_log.append({
            'Column': col, 'DType': str(col_dtype), 'Missing %': round(missing_pct * 100, 2),
            'Strategy Applied': strategy
        })
        
    # Output the decision matrix for review
    if decision_log:
        summary_df = pd.DataFrame(decision_log)
        print("--- Missing Value Handling Strategy Log ---")
        print(summary_df.to_string(index=False))
    else:
        print("No missing values found in the dataset.")
        
    print("-" * 55)
    print(f"Original Shape: {df.shape} | Cleaned Shape: {cleaned_df.shape}")
    
    return cleaned_df

# --- How to use it ---
# cleaned_df = handle_missing_values(df)


# Identify outlier columns

def get_outlier_column_names(df):
    outlier_columns = []
    
    # Filter for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    for col in numeric_df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        # Calculate Q1, Q3, and IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # .any() stops checking once it finds the first outlier, making it faster
        has_outliers = ((series < lower_bound) | (series > upper_bound)).any()
        
        if has_outliers:
            outlier_columns.append(col)
            
    return outlier_columns

# --- Example Usage ---
# df = pd.read_csv('loan_data.csv')
# columns_with_outliers = get_outlier_column_names(df)
# print(columns_with_outliers)


# Handling outlers
def clean_and_handle_outliers(df):
    """
    Identifies column types, calculates outlier distributions, 
    selects an automated strategy (Removal, Clipping, or Winsorization), 
    and applies the correction.
    """
    # Create a copy to prevent modifying the original dataframe in-place
    cleaned_df = df.copy()
    decision_log = []
    
    for col in cleaned_df.columns:
        col_dtype = cleaned_df[col].dtype
        
        # Step 1: Identify Nature/Format
        if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            decision_log.append({
                'Column': col, 'DType': str(col_dtype), 'Outlier %': 0, 
                'Strategy Applied': 'Ignored (Categorical/Text)'
            })
            continue
            
        series = cleaned_df[col].dropna()
        if len(series) == 0:
            continue
            
        # Step 2: Calculate IQR and Outlier Metrics
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers_mask.sum()
        outlier_pct = (outlier_count / len(series)) * 100
        
        if outlier_count == 0:
            decision_log.append({
                'Column': col, 'DType': str(col_dtype), 'Outlier %': 0.0, 
                'Strategy Applied': 'None (No Outliers Found)'
            })
            continue
            
        # Step 3: Determine Strategy Based on Data Distribution
        skewness = series.skew()
        
        if outlier_pct <= 1.5:
            # Low volume of outliers: Safe to remove rows without heavy data loss
            strategy = 'Removal'
        elif abs(skewness) > 1.5:
            # High skewness (long tails): Winsorization protects the distribution shape 
            # by capping at strict percentiles rather than rigid IQR bounds.
            strategy = 'Winsorization (5th-95th Percentile)'
        else:
            # Moderate/Normal distribution with many outliers: Clip exactly at IQR bounds
            strategy = 'Clipping (IQR Bounds)'
            
        decision_log.append({
            'Column': col, 'DType': str(col_dtype), 
            'Outlier %': round(outlier_pct, 2), 
            'Strategy Applied': strategy
        })
        
        # Step 4: Apply the Selected Strategy
        if strategy == 'Removal':
            # Keep rows within bounds OR where data was originally NaN (to preserve nulls for separate imputation)
            valid_rows_mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound) | cleaned_df[col].isna()
            cleaned_df = cleaned_df[valid_rows_mask]
            
        elif strategy == 'Clipping (IQR Bounds)':
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif strategy == 'Winsorization (5th-95th Percentile)':
            p05 = series.quantile(0.05)
            p95 = series.quantile(0.95)
            cleaned_df[col] = cleaned_df[col].clip(lower=p05, upper=p95)
            
    # Output the decision matrix for review
    summary_df = pd.DataFrame(decision_log)
    print("--- Outlier Handling Strategy Log ---")
    print(summary_df.to_string(index=False))
    print("-" * 35)
    print(f"Original Row Count: {len(df)} | Cleaned Row Count: {len(cleaned_df)}")
    
    return cleaned_df

# --- Execution on Portfolio Data ---
# df = pd.read_csv('loan_data.csv')
# cleaned_df = clean_and_handle_outliers(df)

# Basic EDA

def perform_borrower_eda(df):
    """
    Performs complete Exploratory Data Analysis on the loan dataset to 
    identify trends, feature relationships, and reliable borrower profiles.
    """
    # 1. Feature Engineering
    # Calculate Loan-to-Income (LTI) ratio to measure borrower leverage
    df['loan_to_income'] = df['loan_amount'] / df['income']
    
    # 2. Key Metrics Calculation
    total_loans = len(df)
    default_rate = df['default'].mean()
    avg_loan = df['loan_amount'].mean()
    
    metrics = {
        "Total Borrowers": total_loans,
        "Overall Default Rate": f"{default_rate * 100:.2f}%",
        "Average Loan Amount": f"${avg_loan:,.2f}"
    }
    
    # 3. Visualizations
    fig = plt.figure(figsize=(18, 12))
    sns.set_theme(style="whitegrid")
    
    # Plot A: Feature Correlation Matrix
    plt.subplot(2, 3, 1)
    corr = df[['age', 'income', 'loan_amount', 'credit_score', 'loan_to_income', 'default']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.title('Feature Correlation Matrix')
    
    # Plot B: Loan Amount by Default Status
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='default', y='loan_amount', palette='Set2')
    plt.title('Loan Amount vs Default (0=Good, 1=Bad)')
    
    # Plot C: Credit Score Distribution
    plt.subplot(2, 3, 3)
    sns.violinplot(data=df, x='default', y='credit_score', palette='muted')
    plt.title('Credit Score vs Default')
    
    # Plot D: Age Distribution
    plt.subplot(2, 3, 4)
    sns.boxplot(data=df, x='default', y='age', palette='pastel')
    plt.title('Age vs Default')
    
    # Plot E: Loan-to-Income Ratio
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='default', y='loan_to_income', palette='dark:salmon_r')
    plt.title('Loan-to-Income (LTI) Ratio vs Default')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Two-Dimensional Risk Profiling (Credit Score + LTI)
    # Segment data into tiers to identify reliable vs unreliable borrowers
    df_clean = df.dropna(subset=['credit_score', 'loan_to_income']).copy()
    
    df_clean['lti_tier'] = pd.qcut(df_clean['loan_to_income'], q=3, labels=['Low LTI', 'Medium LTI', 'High LTI'])
    df_clean['credit_tier'] = pd.qcut(df_clean['credit_score'], q=3, labels=['Poor', 'Fair', 'Good'])
    
    risk_matrix = df_clean.groupby(['credit_tier', 'lti_tier'], observed=False)['default'].agg(['count', 'mean']).reset_index()
    risk_matrix.rename(columns={'count': 'Total Borrowers', 'mean': 'Default Rate'}, inplace=True)
    risk_matrix['Default Rate'] = (risk_matrix['Default Rate'] * 100).round(2).astype(str) + '%'
    
    return metrics, risk_matrix

# Execute the function
# df = pd.read_csv('loan_data.csv')
# metrics, risk_matrix = perform_borrower_eda(df)