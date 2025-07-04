"""
Data Comparison Utility
Compares processed data from raw files with the existing xgboost_ready_dataset.csv
to validate that preprocessing produces identical results
"""

import pandas as pd
import numpy as np
import os

def compare_datasets(processed_df, reference_csv_path="Data/xgboost_ready_dataset.csv"):
    """
    Compare processed dataframe with reference CSV to ensure identical results
    
    Args:
        processed_df: DataFrame processed from raw data
        reference_csv_path: Path to reference xgboost_ready_dataset.csv
    
    Returns:
        dict: Comparison results with detailed analysis
    """
    
    if not os.path.exists(reference_csv_path):
        return {
            'success': False,
            'message': f"Reference file not found: {reference_csv_path}",
            'differences': []
        }
    
    try:
        # Load reference dataset
        reference_df = pd.read_csv(reference_csv_path)
        
        # Basic shape comparison
        differences = []
        
        if processed_df.shape != reference_df.shape:
            differences.append(f"Shape mismatch: Processed {processed_df.shape} vs Reference {reference_df.shape}")
        
        # Column comparison
        processed_cols = set(processed_df.columns)
        reference_cols = set(reference_df.columns)
        
        missing_in_processed = reference_cols - processed_cols
        extra_in_processed = processed_cols - reference_cols
        
        if missing_in_processed:
            differences.append(f"Missing columns in processed: {list(missing_in_processed)}")
        
        if extra_in_processed:
            differences.append(f"Extra columns in processed: {list(extra_in_processed)}")
        
        # If shapes and columns match, compare data values
        if processed_df.shape == reference_df.shape and processed_cols == reference_cols:
            
            # Sort both dataframes by common identifier columns for proper comparison
            sort_cols = ['Bulan', 'Produk_Variasi'] if all(col in processed_df.columns for col in ['Bulan', 'Produk_Variasi']) else []
            
            if sort_cols:
                processed_sorted = processed_df.sort_values(sort_cols).reset_index(drop=True)
                reference_sorted = reference_df.sort_values(sort_cols).reset_index(drop=True)
            else:
                processed_sorted = processed_df.copy()
                reference_sorted = reference_df.copy()
            
            # Compare each column
            for col in reference_sorted.columns:
                if col in processed_sorted.columns:
                    
                    # Handle different data types
                    if processed_sorted[col].dtype != reference_sorted[col].dtype:
                        differences.append(f"Data type mismatch in '{col}': {processed_sorted[col].dtype} vs {reference_sorted[col].dtype}")
                        continue
                    
                    # For numeric columns, use approximate comparison
                    if pd.api.types.is_numeric_dtype(processed_sorted[col]):
                        if not np.allclose(processed_sorted[col], reference_sorted[col], 
                                         rtol=1e-10, atol=1e-10, equal_nan=True):
                            
                            # Calculate differences for numeric columns
                            diff_mask = ~np.isclose(processed_sorted[col], reference_sorted[col], 
                                                  rtol=1e-10, atol=1e-10, equal_nan=True)
                            num_differences = diff_mask.sum()
                            
                            if num_differences > 0:
                                max_diff = np.abs(processed_sorted[col] - reference_sorted[col]).max()
                                differences.append(f"Numeric differences in '{col}': {num_differences} rows differ, max diff: {max_diff}")
                    
                    # For string/object columns, use exact comparison
                    else:
                        # Handle NaN values in string comparison
                        processed_filled = processed_sorted[col].fillna('__NULL__')
                        reference_filled = reference_sorted[col].fillna('__NULL__')
                        
                        if not processed_filled.equals(reference_filled):
                            diff_count = (processed_filled != reference_filled).sum()
                            differences.append(f"String differences in '{col}': {diff_count} rows differ")
        
        # Determine success
        success = len(differences) == 0
        
        if success:
            message = "✅ PERFECT MATCH! Processed data is identical to reference dataset."
        else:
            message = f"❌ Found {len(differences)} difference(s) between datasets."
        
        return {
            'success': success,
            'message': message,
            'differences': differences,
            'processed_shape': processed_df.shape,
            'reference_shape': reference_df.shape,
            'processed_columns': len(processed_df.columns),
            'reference_columns': len(reference_df.columns)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Error during comparison: {str(e)}",
            'differences': [str(e)]
        }

def detailed_column_comparison(processed_df, reference_csv_path="Data/xgboost_ready_dataset.csv"):
    """
    Perform detailed column-by-column comparison
    """
    if not os.path.exists(reference_csv_path):
        return None
    
    reference_df = pd.read_csv(reference_csv_path)
    
    # Sort both dataframes
    sort_cols = ['Bulan', 'Produk_Variasi'] if all(col in processed_df.columns for col in ['Bulan', 'Produk_Variasi']) else []
    
    if sort_cols:
        processed_sorted = processed_df.sort_values(sort_cols).reset_index(drop=True)
        reference_sorted = reference_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        processed_sorted = processed_df.copy()
        reference_sorted = reference_df.copy()
    
    column_results = {}
    
    common_columns = set(processed_sorted.columns) & set(reference_sorted.columns)
    
    for col in common_columns:
        proc_col = processed_sorted[col]
        ref_col = reference_sorted[col]
        
        result = {
            'data_type_match': proc_col.dtype == ref_col.dtype,
            'processed_dtype': str(proc_col.dtype),
            'reference_dtype': str(ref_col.dtype),
            'null_count_processed': proc_col.isnull().sum(),
            'null_count_reference': ref_col.isnull().sum(),
        }
        
        if pd.api.types.is_numeric_dtype(proc_col) and pd.api.types.is_numeric_dtype(ref_col):
            result['is_numeric'] = True
            result['values_match'] = np.allclose(proc_col, ref_col, rtol=1e-10, atol=1e-10, equal_nan=True)
            
            if not result['values_match']:
                diff = np.abs(proc_col - ref_col)
                result['max_difference'] = diff.max()
                result['mean_difference'] = diff.mean()
                result['different_rows'] = (~np.isclose(proc_col, ref_col, rtol=1e-10, atol=1e-10, equal_nan=True)).sum()
        else:
            result['is_numeric'] = False
            proc_filled = proc_col.fillna('__NULL__')
            ref_filled = ref_col.fillna('__NULL__')
            result['values_match'] = proc_filled.equals(ref_filled)
            
            if not result['values_match']:
                result['different_rows'] = (proc_filled != ref_filled).sum()
        
        column_results[col] = result
    
    return column_results

def save_comparison_report(comparison_result, column_comparison=None, output_path="comparison_report.txt"):
    """
    Save detailed comparison report to file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DATASET COMPARISON REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Overall Result: {comparison_result['message']}\n")
        f.write(f"Success: {comparison_result['success']}\n\n")
        
        f.write(f"Processed Shape: {comparison_result.get('processed_shape', 'N/A')}\n")
        f.write(f"Reference Shape: {comparison_result.get('reference_shape', 'N/A')}\n")
        f.write(f"Processed Columns: {comparison_result.get('processed_columns', 'N/A')}\n")
        f.write(f"Reference Columns: {comparison_result.get('reference_columns', 'N/A')}\n\n")
        
        if comparison_result['differences']:
            f.write("DIFFERENCES FOUND:\n")
            f.write("-"*30 + "\n")
            for i, diff in enumerate(comparison_result['differences'], 1):
                f.write(f"{i}. {diff}\n")
            f.write("\n")
        
        if column_comparison:
            f.write("DETAILED COLUMN COMPARISON:\n")
            f.write("-"*40 + "\n")
            
            for col, details in column_comparison.items():
                f.write(f"\nColumn: {col}\n")
                f.write(f"  Data Types Match: {details['data_type_match']}\n")
                f.write(f"  Processed Type: {details['processed_dtype']}\n")
                f.write(f"  Reference Type: {details['reference_dtype']}\n")
                f.write(f"  Values Match: {details['values_match']}\n")
                
                if not details['values_match'] and 'different_rows' in details:
                    f.write(f"  Different Rows: {details['different_rows']}\n")
                    
                    if details.get('is_numeric') and 'max_difference' in details:
                        f.write(f"  Max Difference: {details['max_difference']}\n")
                        f.write(f"  Mean Difference: {details['mean_difference']}\n")
    
    print(f"Comparison report saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    print("Dataset Comparison Utility")
    print("Use this to compare processed data with reference xgboost_ready_dataset.csv")
    print("Import this module and use compare_datasets() function in your processing pipeline.")
