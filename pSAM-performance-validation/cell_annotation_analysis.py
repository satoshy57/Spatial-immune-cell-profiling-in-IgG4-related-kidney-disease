import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
from pathlib import Path
import glob

def load_data(ground_truth_path, annotation_path):
    """
    Load ground truth and annotation data
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth CSV file
    annotation_path : str
        Path to the annotation CSV file
        
    Returns:
    --------
    gt_df : pandas.DataFrame
        Ground truth data
    anno_df : pandas.DataFrame
        Annotation data
    """
    # Load ground truth data
    gt_df = pd.read_csv(ground_truth_path)
    
    # Load annotation data
    anno_df = pd.read_csv(annotation_path)
    
    return gt_df, anno_df

def preprocess_ground_truth(gt_df):
    """
    Preprocess ground truth data to handle duplicate cells based on TAG priority
    
    Parameters:
    -----------
    gt_df : pandas.DataFrame
        Ground truth data
        
    Returns:
    --------
    deduplicated_gt_df : pandas.DataFrame
        Processed ground truth data with duplicates handled based on TAG priority
    tag_counts_after : pandas.DataFrame
        Count of cells in each MemCatName2 by TAG type after deduplication
    tag_counts_before : pandas.DataFrame
        Count of cells in each MemCatName2 by TAG type before deduplication
    tag_label_counts_after : pandas.DataFrame
        Count of cells in each label_name by TAG type after deduplication
    tag_label_counts_before : pandas.DataFrame
        Count of cells in each label_name by TAG type before deduplication
    """
    # Check if TAG column exists
    if 'TAG' not in gt_df.columns:
        print("Warning: 'TAG' column not found in ground truth data. Skipping TAG-based deduplication.")
        # Create a unique identifier for each cell based on CompName and CellID
        if 'CompName' in gt_df.columns and 'CellID' in gt_df.columns:
            gt_df['Cell_Identifier'] = gt_df['CompName'] + '_' + gt_df['CellID'].astype(str)
        else:
            raise ValueError("Ground truth data must contain 'CompName' and 'CellID' columns")
        
        # No deduplication or TAG analysis needed
        return gt_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Create a unique identifier for each cell
    if 'CellCentroidRow' in gt_df.columns and 'CellCentroidCol' in gt_df.columns:
        gt_df['Cell_Identifier'] = gt_df['CompName'] + '_' + gt_df['CellCentroidRow'].astype(str) + '_' + gt_df['CellCentroidCol'].astype(str)
    else:
        print("Warning: Cell centroid columns not found. Using CompName and CellID to create identifier.")
        gt_df['Cell_Identifier'] = gt_df['CompName'] + '_' + gt_df['CellID'].astype(str)
    
    # Identify duplicate cells
    duplicated_cells = gt_df[gt_df.duplicated(subset=['Cell_Identifier'], keep=False)]
    
    # Add TAG_Type classification for analysis
    gt_df['TAG_Type'] = gt_df['TAG'].apply(lambda x: 
                                        'cut_samples-TUBULES' if isinstance(x, str) and 'cut_samples-TUBULES' in x else
                                        ('cut_samples-MPO' if isinstance(x, str) and 'cut_samples-MPO' in x else
                                        ('cut_samples' if isinstance(x, str) and 'cut_samples' in x else 'other')))
    
    # Create counts before deduplication
    # MemCatName2 counts before deduplication
    tag_counts_before = pd.crosstab(gt_df['MemCatName2'], gt_df['TAG_Type'])
    tag_counts_before.loc['Total'] = tag_counts_before.sum()
    tag_counts_before['Total'] = tag_counts_before.sum(axis=1)
    
    # Check if 'label_name' column exists
    if 'label_name' not in gt_df.columns:
        print("Warning: 'label_name' column not found in ground truth data. Using MemCatName2 for label counts.")
        tag_label_counts_before = tag_counts_before.copy()
        # If no duplicates, both before and after tables are the same
        if len(duplicated_cells) == 0:
            print("No duplicate cells found in ground truth data.")
            # Remove temporary column
            gt_df = gt_df.drop(['TAG_Type'], axis=1)
            return gt_df, tag_counts_before, tag_counts_before, tag_label_counts_before, tag_label_counts_before
    else:
        # label_name counts before deduplication
        tag_label_counts_before = pd.crosstab(gt_df['label_name'], gt_df['TAG_Type'])
        tag_label_counts_before.loc['Total'] = tag_label_counts_before.sum()
        tag_label_counts_before['Total'] = tag_label_counts_before.sum(axis=1)
    
    # If no duplicates, both before and after tables are the same
    if len(duplicated_cells) == 0:
        print("No duplicate cells found in ground truth data.")
        # Remove temporary column
        gt_df = gt_df.drop(['TAG_Type'], axis=1)
        return gt_df, tag_counts_before, tag_counts_before, tag_label_counts_before, tag_label_counts_before
    
    # Define TAG priorities
    def get_tag_priority(tag):
        if isinstance(tag, str):
            if 'cut_samples-TUBULES' in tag:
                return 1
            elif 'cut_samples-MPO' in tag:
                return 2
            elif 'cut_samples' in tag:
                return 3
            else:
                return 4
        else:
            return 5
    
    # Add priority column
    gt_df['TAG_Priority'] = gt_df['TAG'].apply(get_tag_priority)
    
    # Sort by Cell_Identifier and TAG_Priority
    gt_df = gt_df.sort_values(['Cell_Identifier', 'TAG_Priority'])
    
    # Keep the highest priority (lowest value) entry for each Cell_Identifier
    deduplicated_gt_df = gt_df.drop_duplicates(subset=['Cell_Identifier'], keep='first')
    
    print(f"Found {len(duplicated_cells)} duplicate cell records in ground truth data.")
    print(f"Removed {len(gt_df) - len(deduplicated_gt_df)} duplicates based on TAG priority.")
    
    # Copy TAG_Type to deduplicated dataframe for crosstab
    deduplicated_gt_df['TAG_Type'] = deduplicated_gt_df['TAG'].apply(lambda x: 
                                      'cut_samples-TUBULES' if isinstance(x, str) and 'cut_samples-TUBULES' in x else
                                      ('cut_samples-MPO' if isinstance(x, str) and 'cut_samples-MPO' in x else
                                       ('cut_samples' if isinstance(x, str) and 'cut_samples' in x else 'other')))
    
    # Get counts after deduplication
    # MemCatName2 counts after deduplication
    tag_counts_after = pd.crosstab(deduplicated_gt_df['MemCatName2'], deduplicated_gt_df['TAG_Type'])
    tag_counts_after.loc['Total'] = tag_counts_after.sum()
    tag_counts_after['Total'] = tag_counts_after.sum(axis=1)
    
    # label_name counts after deduplication
    if 'label_name' not in deduplicated_gt_df.columns:
        tag_label_counts_after = tag_counts_after.copy()
    else:
        tag_label_counts_after = pd.crosstab(deduplicated_gt_df['label_name'], deduplicated_gt_df['TAG_Type'])
        tag_label_counts_after.loc['Total'] = tag_label_counts_after.sum()
        tag_label_counts_after['Total'] = tag_label_counts_after.sum(axis=1)
    
    # Drop the temporary columns
    deduplicated_gt_df = deduplicated_gt_df.drop(['TAG_Priority', 'TAG_Type'], axis=1)
    
    return deduplicated_gt_df, tag_counts_after, tag_counts_before, tag_label_counts_after, tag_label_counts_before


def preprocess_data(gt_df, anno_df):
    """
    Preprocess data for analysis
    
    Parameters:
    -----------
    gt_df : pandas.DataFrame
        Ground truth data
    anno_df : pandas.DataFrame
        Annotation data
        
    Returns:
    --------
    merged_df : pandas.DataFrame
        Merged data with ground truth and annotation information
    """
    # Ensure CompName and CellID are in the correct format
    if 'CompName' in gt_df.columns and 'CellID' in gt_df.columns:
        # Create a unique identifier for each cell
        gt_df['Cell_Identifier'] = gt_df['CompName'] + '_' + gt_df['CellID'].astype(str)
    else:
        raise ValueError("Ground truth data must contain 'CompName' and 'CellID' columns")
    
    if 'CompName' in anno_df.columns and 'CellID' in anno_df.columns:
        # Create a unique identifier for each cell
        anno_df['Cell_Identifier'] = anno_df['CompName'] + '_' + anno_df['CellID'].astype(str)
    else:
        raise ValueError("Annotation data must contain 'CompName' and 'CellID' columns")
    
    # Merge ground truth and annotation data
    merged_df = pd.merge(
        gt_df[['Cell_Identifier', 'MemCatName2']], 
        anno_df[['Cell_Identifier', 'MemCatName2']], 
        on='Cell_Identifier', 
        how='inner', 
        suffixes=('_gt', '_anno')
    )
    
    return merged_df

def calculate_annotation_stats(merged_df):
    """
    Calculate annotation statistics
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged data with ground truth and annotation information
        
    Returns:
    --------
    stats_df : pandas.DataFrame
        Statistics of annotation performance per MemCatName2
    """
    # Get unique cell types in ground truth
    cell_types = merged_df['MemCatName2_gt'].unique()
    
    # Initialize lists to store results
    cell_type_list = []
    total_cells_list = []
    correct_cells_list = []
    correct_percentage_list = []
    wrong_percentage_list = []
    
    # Calculate statistics for each cell type
    for cell_type in cell_types:
        # Get cells of current type from ground truth
        cells_of_type = merged_df[merged_df['MemCatName2_gt'] == cell_type]
        
        # Count total cells of current type
        total_cells = len(cells_of_type)
        
        # Count correctly annotated cells
        correct_cells = len(cells_of_type[cells_of_type['MemCatName2_gt'] == cells_of_type['MemCatName2_anno']])
        
        # Calculate percentages
        correct_percentage = (correct_cells / total_cells) * 100 if total_cells > 0 else 0
        wrong_percentage = 100 - correct_percentage
        
        # Append results to lists
        cell_type_list.append(cell_type)
        total_cells_list.append(total_cells)
        correct_cells_list.append(correct_cells)
        correct_percentage_list.append(correct_percentage)
        wrong_percentage_list.append(wrong_percentage)
    
    # Create DataFrame with results
    stats_df = pd.DataFrame({
        'Cell_Type': cell_type_list,
        'Total_Cells': total_cells_list,
        'Correct_Cells': correct_cells_list,
        'Correct_Percentage': correct_percentage_list,
        'Wrong_Percentage': wrong_percentage_list
    })
    
    return stats_df

def calculate_metrics(merged_df, include_cell_types=None):
    """
    Calculate precision, recall, and F1 score
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged data with ground truth and annotation information
    include_cell_types : list, optional
        List of cell types to include in the analysis. If None, all cell types are included.
        
    Returns:
    --------
    metrics_df : pandas.DataFrame
        Precision, recall, and F1 score per cell type
    """
    # Filter by selected cell types if specified
    if include_cell_types is not None:
        merged_df = merged_df[
            (merged_df['MemCatName2_gt'].isin(include_cell_types)) | 
            (merged_df['MemCatName2_anno'].isin(include_cell_types))
        ]
    
    # Get unique cell types, filtering out NaN values
    gt_types = [x for x in merged_df['MemCatName2_gt'].unique() if pd.notna(x)]
    anno_types = [x for x in merged_df['MemCatName2_anno'].unique() if pd.notna(x)]
    cell_types = sorted(list(set(gt_types) | set(anno_types)))
    
    # Filter cell types if specified
    if include_cell_types is not None:
        cell_types = [ct for ct in cell_types if ct in include_cell_types]
    
    # Create mapping for cell types to integers
    cell_type_to_int = {cell_type: i for i, cell_type in enumerate(cell_types)}
    int_to_cell_type = {i: cell_type for i, cell_type in enumerate(cell_types)}
    
    # Convert cell types to integers, handling NaN values
    y_true = []
    for cell_type in merged_df['MemCatName2_gt']:
        if pd.isna(cell_type):
            y_true.append(-2)  # Special code for NaN
        elif cell_type in cell_type_to_int:
            y_true.append(cell_type_to_int[cell_type])
        else:
            y_true.append(-1)  # Unknown cell type
    
    y_pred = []
    for cell_type in merged_df['MemCatName2_anno']:
        if pd.isna(cell_type):
            y_pred.append(-2)  # Special code for NaN
        elif cell_type in cell_type_to_int:
            y_pred.append(cell_type_to_int[cell_type])
        else:
            y_pred.append(-1)  # Unknown cell type
            
    # Filter out NaN values and non-included cell types for metric calculation
    valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                    if true != -2 and pred != -2 and true != -1 and pred != -1]
    filtered_y_true = [y_true[i] for i in valid_indices]
    filtered_y_pred = [y_pred[i] for i in valid_indices]
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_y_true, 
        filtered_y_pred, 
        labels=range(len(cell_types)),
        zero_division=0  # Handle division by zero
    )
    
    # Create DataFrame with results
    metrics_df = pd.DataFrame({
        'Cell_Type': [int_to_cell_type[i] for i in range(len(cell_types))],
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Support': support
    })
    
    # Calculate micro-average metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        filtered_y_true, filtered_y_pred, average='micro', zero_division=0
    )
    
    # Calculate macro-average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        filtered_y_true, filtered_y_pred, average='macro', zero_division=0
    )
    
    # Calculate weighted-average metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        filtered_y_true, filtered_y_pred, average='weighted', zero_division=0
    )
    
    # Create DataFrame with average metrics
    avg_metrics_df = pd.DataFrame({
        'Cell_Type': ['micro-avg', 'macro-avg', 'weighted-avg'],
        'Precision': [micro_precision, macro_precision, weighted_precision],
        'Recall': [micro_recall, macro_recall, weighted_recall],
        'F1_Score': [micro_f1, macro_f1, weighted_f1],
        'Support': [sum(support), np.nan, sum(support)]
    })
    
    # Concatenate metrics DataFrame and average metrics DataFrame
    metrics_df = pd.concat([metrics_df, avg_metrics_df], ignore_index=True)
    
    return metrics_df

def create_confusion_matrix(merged_df, include_cell_types=None):
    """
    Create confusion matrix
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged data with ground truth and annotation information
    include_cell_types : list, optional
        List of cell types to include in the analysis. If None, all cell types are included.
        
    Returns:
    --------
    cm : numpy.ndarray
        Confusion matrix
    cell_types : list
        List of cell types
    """
    # Filter by selected cell types if specified
    if include_cell_types is not None:
        merged_df = merged_df[
            (merged_df['MemCatName2_gt'].isin(include_cell_types)) & 
            (merged_df['MemCatName2_anno'].isin(include_cell_types))
        ]
    
    # Get unique cell types, filtering out NaN values
    gt_types = [x for x in merged_df['MemCatName2_gt'].unique() if pd.notna(x)]
    anno_types = [x for x in merged_df['MemCatName2_anno'].unique() if pd.notna(x)]
    cell_types = sorted(list(set(gt_types) | set(anno_types)))
    
    # Filter cell types if specified
    if include_cell_types is not None:
        cell_types = [ct for ct in cell_types if ct in include_cell_types]
    
    # Create mapping for cell types to integers
    cell_type_to_int = {cell_type: i for i, cell_type in enumerate(cell_types)}
    
    # Filter out rows with NaN values or non-included cell types
    valid_df = merged_df.dropna(subset=['MemCatName2_gt', 'MemCatName2_anno'])
    if include_cell_types is not None:
        valid_df = valid_df[
            (valid_df['MemCatName2_gt'].isin(include_cell_types)) & 
            (valid_df['MemCatName2_anno'].isin(include_cell_types))
        ]
    
    if valid_df.empty:
        # If no valid data, return empty confusion matrix
        return np.zeros((len(cell_types), len(cell_types))), cell_types
    
    # Convert cell types to integers
    y_true = [cell_type_to_int[cell_type] for cell_type in valid_df['MemCatName2_gt'] 
              if cell_type in cell_type_to_int]
    y_pred = [cell_type_to_int[cell_type] for cell_type in valid_df['MemCatName2_anno'] 
              if cell_type in cell_type_to_int]
    
    # Ensure y_true and y_pred have the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(cell_types)))
    
    return cm, cell_types

def plot_confusion_matrix(cm, cell_types, title='Confusion Matrix', normalize=True):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    cell_types : list
        List of cell types
    title : str
        Title of the plot
    normalize : bool
        Whether to normalize the confusion matrix
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Normalize confusion matrix if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set ticks and labels
    tick_marks = np.arange(len(cell_types))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(cell_types, rotation=90)
    ax.set_yticklabels(cell_types)
    
    # Add labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                ax.text(j, i, f'{cm[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, f'{int(cm[i, j])}',
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    return fig

def plot_metrics(metrics_df):
    """
    Plot precision, recall, and F1 score
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Precision, recall, and F1 score per cell type
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Filter out average metrics and rows with NaN values
    cell_metrics_df = metrics_df[~metrics_df['Cell_Type'].isin(['micro-avg', 'macro-avg', 'weighted-avg'])]
    cell_metrics_df = cell_metrics_df.dropna(subset=['Precision', 'Recall', 'F1_Score'])
    
    if cell_metrics_df.empty:
        # If no valid data, create an empty plot with a message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No valid metrics data available", 
                ha='center', va='center', fontsize=18)
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 Score per Cell Type')
        plt.tight_layout()
        return fig
    
    # Sort by F1 score in descending order
    cell_metrics_df = cell_metrics_df.sort_values('F1_Score', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar positions
    x = np.arange(len(cell_metrics_df))
    width = 0.25
    
    # Create bars
    ax.bar(x - width, cell_metrics_df['Precision'], width, label='Precision')
    ax.bar(x, cell_metrics_df['Recall'], width, label='Recall')
    ax.bar(x + width, cell_metrics_df['F1_Score'], width, label='F1 Score')
    
    # Add labels and title
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 Score per Cell Type')
    
    # Set ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(cell_metrics_df['Cell_Type'], rotation=90)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def compare_annotations(ground_truth_df, annotation_paths, names=None, include_cell_types=None):
    """
    Compare multiple annotation settings
    
    Parameters:
    -----------
    ground_truth_df : pandas.DataFrame
        Processed ground truth data
    annotation_paths : list
        List of paths to annotation CSV files
    names : list
        List of names for annotation settings
    include_cell_types : list, optional
        List of cell types to include in the analysis. If None, all cell types are included.
        
    Returns:
    --------
    comparison_df : pandas.DataFrame
        Comparison of precision, recall, and F1 score for different annotation settings
    """
    # Set default names if not provided
    if names is None:
        names = [f'Setting {i+1}' for i in range(len(annotation_paths))]
    
    # Ensure names and annotation_paths have the same length
    if len(names) != len(annotation_paths):
        print(f"Warning: Number of names ({len(names)}) does not match number of annotation paths ({len(annotation_paths)})")
        # Adjust names list to match annotation_paths length
        if len(names) < len(annotation_paths):
            names.extend([f'Setting {i+1}' for i in range(len(names), len(annotation_paths))])
        else:
            names = names[:len(annotation_paths)]
            
    # Try to sort settings based on numeric value in 'memth X' format
    sorted_indices = list(range(len(names)))
    try:
        # Check if all settings follow the pattern 'memth X'
        if all('memth' in str(name) for name in names):
            # Extract the numeric part and sort based on that
            name_values = [int(str(name).split('memth ')[1]) for name in names]
            sorted_indices = sorted(range(len(name_values)), key=lambda i: name_values[i])
            # Reorder names and annotation_paths
            names = [names[i] for i in sorted_indices]
            annotation_paths = [annotation_paths[i] for i in sorted_indices]
    except:
        # If any error occurs, use the original order
        pass
    
    # Use the provided ground truth DataFrame
    gt_df = ground_truth_df
    
    # Initialize list to store metrics DataFrames
    metrics_dfs = []
    
    # Process each annotation setting
    for i, annotation_path in enumerate(annotation_paths):
        try:
            print(f"Calculating metrics for {names[i]}...")
            # Load annotation data
            anno_df = pd.read_csv(annotation_path)
            
            # Preprocess data
            merged_df = preprocess_data(gt_df, anno_df)
            
            # Calculate metrics with optional cell type filtering
            metrics_df = calculate_metrics(merged_df, include_cell_types)
            
            # Add name column
            metrics_df['Setting'] = names[i]
            
            # Append to list
            metrics_dfs.append(metrics_df)
        except Exception as e:
            print(f"Error calculating metrics for {names[i]}: {str(e)}")
            # Create empty DataFrame with appropriate columns
            empty_metrics = pd.DataFrame(columns=['Cell_Type', 'Precision', 'Recall', 'F1_Score', 'Support'])
            empty_metrics['Setting'] = names[i]
            metrics_dfs.append(empty_metrics)
    
    # Concatenate metrics DataFrames
    all_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
    
    return all_metrics_df

def plot_comparison(comparison_df, metric='F1_Score'):
    """
    Plot comparison of metrics for different annotation settings
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        Comparison of precision, recall, and F1 score for different annotation settings
    metric : str
        Metric to plot (Precision, Recall, or F1_Score)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get unique cell types (excluding average metrics and NaN values)
    valid_cell_types = comparison_df[
        (~comparison_df['Cell_Type'].isin(['micro-avg', 'macro-avg', 'weighted-avg'])) & 
        (pd.notna(comparison_df['Cell_Type']))
    ]['Cell_Type'].unique()
    
    # Get settings and sort them correctly if they follow the pattern 'memth X'
    settings = comparison_df['Setting'].unique()
    
    # Try to sort settings based on numeric value in 'memth X' format
    try:
        # Check if all settings follow the pattern 'memth X'
        if all('memth' in str(setting) for setting in settings):
            # Extract the numeric part and sort based on that
            settings_dict = {setting: int(str(setting).split('memth ')[1]) for setting in settings}
            settings = sorted(settings, key=lambda x: settings_dict[x])
    except:
        # If any error occurs, use the original settings order
        pass
    
    # Filter out rows with NaN metric values
    filtered_df = comparison_df.dropna(subset=[metric])
    
    # If there's no valid data, create an empty plot with a message
    if filtered_df.empty or len(valid_cell_types) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f"No valid {metric} data available for comparison", 
                ha='center', va='center', fontsize=18)
        ax.set_xlabel('Annotation Setting')
        ax.set_ylabel('Cell Type')
        ax.set_title(f'{metric} per Cell Type and Annotation Setting')
        plt.tight_layout()
        return fig
    
    # Create pivot table
    pivot_df = filtered_df[filtered_df['Cell_Type'].isin(valid_cell_types)].pivot(
        index='Cell_Type', columns='Setting', values=metric
    )
    
    # Reorder columns based on sorted settings
    if all(setting in pivot_df.columns for setting in settings):
        pivot_df = pivot_df[settings]
    
    # Fill NaN values with 0 for sorting
    sort_df = pivot_df.fillna(0)
    
    # Sort by mean score in descending order
    pivot_df = pivot_df.reindex(sort_df.mean(axis=1).sort_values(ascending=False).index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with fixed colorbar range from 0.50 to 1.00
    sns.heatmap(pivot_df, annot=False, cmap='YlGnBu', fmt='.2f', 
                linewidths=0.5, ax=ax, cbar_kws={'label': metric},
                vmin=0.50, vmax=1.00)
    
    # Add labels and title
    ax.set_xlabel('Annotation Setting')
    ax.set_ylabel('Cell Type')
    ax.set_title(f'{metric} per Cell Type and Annotation Setting')
    
    plt.tight_layout()
    
    return fig

def plot_average_metrics(comparison_df):
    """
    Plot average metrics with error bars for different annotation settings
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        Comparison of precision, recall, and F1 score for different annotation settings
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get unique settings and sort them correctly if they follow the pattern 'memth X'
    settings = comparison_df['Setting'].unique()
    
    # Try to sort settings based on numeric value in 'memth X' format
    try:
        # Check if all settings follow the pattern 'memth X'
        if all('memth' in str(setting) for setting in settings):
            # Extract the numeric part and sort based on that
            settings_dict = {setting: int(str(setting).split('memth ')[1]) for setting in settings}
            settings = sorted(settings, key=lambda x: settings_dict[x])
    except:
        # If any error occurs, use the original settings order
        pass
    
    # Get weighted-average metrics
    weighted_avg_df = comparison_df[comparison_df['Cell_Type'] == 'weighted-avg']
    
    # Reorder the DataFrame to match the sorted settings
    if not weighted_avg_df.empty:
        ordered_df = pd.DataFrame()
        for setting in settings:
            setting_df = weighted_avg_df[weighted_avg_df['Setting'] == setting]
            ordered_df = pd.concat([ordered_df, setting_df])
        weighted_avg_df = ordered_df
    
    # If no weighted average metrics available, try using macro-average
    if weighted_avg_df.empty:
        weighted_avg_df = comparison_df[comparison_df['Cell_Type'] == 'macro-avg']
        
        # Reorder the DataFrame to match the sorted settings
        if not weighted_avg_df.empty:
            ordered_df = pd.DataFrame()
            for setting in settings:
                setting_df = weighted_avg_df[weighted_avg_df['Setting'] == setting]
                ordered_df = pd.concat([ordered_df, setting_df])
            weighted_avg_df = ordered_df
    
    # Calculate standard errors for each metric and setting
    # Filter comparison_df to include only cell-type metrics (not averages)
    cell_metrics_df = comparison_df[~comparison_df['Cell_Type'].isin(['micro-avg', 'macro-avg', 'weighted-avg'])]
    
    # Initialize dictionaries to store standard error values
    precision_std_errors = {}
    recall_std_errors = {}
    f1_std_errors = {}
    
    # Calculate standard errors for each setting
    for setting in settings:
        setting_metrics = cell_metrics_df[cell_metrics_df['Setting'] == setting]
        
        # Skip if no data
        if setting_metrics.empty:
            precision_std_errors[setting] = 0
            recall_std_errors[setting] = 0
            f1_std_errors[setting] = 0
            continue
        
        # Calculate standard deviation and standard error
        precision_values = setting_metrics['Precision'].dropna().values
        recall_values = setting_metrics['Recall'].dropna().values
        f1_values = setting_metrics['F1_Score'].dropna().values
        
        n_precision = len(precision_values)
        n_recall = len(recall_values)
        n_f1 = len(f1_values)
        
        # Calculate standard error (standard deviation / sqrt(n))
        precision_std_errors[setting] = np.std(precision_values) / np.sqrt(n_precision) if n_precision > 0 else 0
        recall_std_errors[setting] = np.std(recall_values) / np.sqrt(n_recall) if n_recall > 0 else 0
        f1_std_errors[setting] = np.std(f1_values) / np.sqrt(n_f1) if n_f1 > 0 else 0
    
    # If still no metrics available, create empty plot with message
    if weighted_avg_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No average metrics data available", 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Annotation Setting')
        ax.set_ylabel('Score')
        ax.set_title('Average Metrics per Annotation Setting with Error Bars')
        plt.tight_layout()
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bar positions
    x = np.arange(len(settings))
    width = 0.25
    
    # Create bars, handling NaN values
    precision = weighted_avg_df['Precision'].fillna(0).values
    recall = weighted_avg_df['Recall'].fillna(0).values
    f1_score = weighted_avg_df['F1_Score'].fillna(0).values
    
    # Get standard errors for each setting
    precision_errors = [precision_std_errors.get(setting, 0) for setting in settings]
    recall_errors = [recall_std_errors.get(setting, 0) for setting in settings]
    f1_errors = [f1_std_errors.get(setting, 0) for setting in settings]
    
    # Create bars with error bars
    ax.bar(x - width, precision, width, label='Precision', color='#1f77b4', alpha=0.8)
    ax.errorbar(x - width, precision, yerr=precision_errors, fmt='none', ecolor='black', capsize=5)
    
    ax.bar(x, recall, width, label='Recall', color='#ff7f0e', alpha=0.8)
    ax.errorbar(x, recall, yerr=recall_errors, fmt='none', ecolor='black', capsize=5)
    
    ax.bar(x + width, f1_score, width, label='F1 Score', color='#2ca02c', alpha=0.8)
    ax.errorbar(x + width, f1_score, yerr=f1_errors, fmt='none', ecolor='black', capsize=5)
    
    # Add labels and title
    ax.set_xlabel('Annotation Setting', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Weighted Average Metrics per Annotation Setting with Error Bars', fontsize=14)
    
    # Set ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.2)
    
    # Add score values above the bars
    for i, v in enumerate(precision):
        ax.text(i - width, v + precision_errors[i] + 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(recall):
        ax.text(i, v + recall_errors[i] + 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(f1_score):
        ax.text(i + width, v + f1_errors[i] + 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add explanation of error bars
    fig.text(0.5, 0.02, 'Error bars represent standard error across all cell types', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

def main(ground_truth_path, annotation_paths, names=None, output_dir=None, include_cell_types=None):
    """
    Main function to run the analysis
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth CSV file
    annotation_paths : list
        List of paths to annotation CSV files
    names : list
        List of names for annotation settings
    output_dir : str
        Path to output directory
    include_cell_types : list, optional
        List of cell types to include in the analysis. If None, all cell types are included.
        
    Returns:
    --------
    None
    """
    # Create output directory if not exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set default names if not provided
    if names is None:
        names = [f'Setting {i+1}' for i in range(len(annotation_paths))]
    
    # Ensure names and annotation_paths have the same length
    if len(names) != len(annotation_paths):
        print(f"Warning: Number of names ({len(names)}) does not match number of annotation paths ({len(annotation_paths)})")
        # Adjust names list to match annotation_paths length
        if len(names) < len(annotation_paths):
            names.extend([f'Setting {i+1}' for i in range(len(names), len(annotation_paths))])
        else:
            names = names[:len(annotation_paths)]
    
    try:
        # Load ground truth data
        gt_df = pd.read_csv(ground_truth_path)
        
        # Preprocess ground truth to handle duplicate cells based on TAG priority
        # Now returns 5 values:
        # 1. Processed ground truth data with duplicates removed
        # 2. MemCatName2 counts after deduplication (tag_counts.csv)
        # 3. MemCatName2 counts before deduplication
        # 4. label_name counts after deduplication
        # 5. label_name counts before deduplication (tag_details_counts.csv)
        processed_gt_df, memcat_counts_after, memcat_counts_before, label_counts_after, label_counts_before = preprocess_ground_truth(gt_df)
        
        # Save MemCatName2 counts after deduplication
        if not memcat_counts_after.empty and output_dir is not None:
            memcat_counts_after.to_csv(os.path.join(output_dir, 'tag_counts.csv'))
            
            # Display MemCatName2 counts after deduplication
            print("\nMemCatName2 counts by TAG type AFTER deduplication:")
            print(memcat_counts_after)
            print("\n")
        
        # Save MemCatName2 counts before deduplication
        if not memcat_counts_before.empty and output_dir is not None:
            memcat_counts_before.to_csv(os.path.join(output_dir, 'tag_counts_before_dedup.csv'))
            
            # Display MemCatName2 counts before deduplication
            print("\nMemCatName2 counts by TAG type BEFORE deduplication:")
            print(memcat_counts_before)
            print("\n")
        
        # Save label_name counts after deduplication
        if not label_counts_after.empty and output_dir is not None:
            label_counts_after.to_csv(os.path.join(output_dir, 'tag_details_counts.csv'))
            
            # Display label_name counts after deduplication
            print("\nlabel_name counts by TAG type AFTER deduplication:")
            print(label_counts_after)
            print("\n")
        
        # Save label_name counts before deduplication
        if not label_counts_before.empty and output_dir is not None:
            label_counts_before.to_csv(os.path.join(output_dir, 'tag_details_counts_before_dedup.csv'))
            
            # Display label_name counts before deduplication
            print("\nlabel_name counts by TAG type BEFORE deduplication:")
            print(label_counts_before)
            print("\n")
        
        # Compare annotations
        print(f"Comparing {len(annotation_paths)} annotation settings...")
        # Use the processed ground truth data instead of loading it again
        comparison_df = compare_annotations(processed_gt_df, annotation_paths, names, include_cell_types)
        
        # Save comparison DataFrame
        if output_dir is not None:
            comparison_df.to_csv(os.path.join(output_dir, 'comparison.csv'), index=False)
        
        # Plot comparison of F1 scores
        print("Generating F1 score comparison plot...")
        fig_f1 = plot_comparison(comparison_df, metric='F1_Score')
        
        # Save F1 comparison plot
        if output_dir is not None:
            fig_f1.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Plot comparison of precision
        print("Generating precision comparison plot...")
        fig_precision = plot_comparison(comparison_df, metric='Precision')
        
        # Save precision comparison plot
        if output_dir is not None:
            fig_precision.savefig(os.path.join(output_dir, 'precision_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Plot comparison of recall
        print("Generating recall comparison plot...")
        fig_recall = plot_comparison(comparison_df, metric='Recall')
        
        # Save recall comparison plot
        if output_dir is not None:
            fig_recall.savefig(os.path.join(output_dir, 'recall_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Plot average metrics with error bars
        print("Generating average metrics plot with error bars...")
        fig_avg = plot_average_metrics(comparison_df)
        
        # Save average metrics plot
        if output_dir is not None:
            fig_avg.savefig(os.path.join(output_dir, 'average_metrics_with_error_bars.png'), dpi=300, bbox_inches='tight')
    except Exception as e:
        import traceback
        print(f"Error in comparison analysis: {str(e)}")
        print(traceback.format_exc())
    
    # Process each annotation setting separately
    for i, annotation_path in enumerate(annotation_paths):
        try:
            print(f"Processing annotation setting {i+1}/{len(annotation_paths)}: {names[i]}...")
            
            # Load annotation data
            anno_df = pd.read_csv(annotation_path)
            
            # Use the already processed ground truth data
            # Preprocess data
            merged_df = preprocess_data(processed_gt_df, anno_df)
            
            # Calculate annotation statistics
            stats_df = calculate_annotation_stats(merged_df)
            
            # Save annotation statistics
            if output_dir is not None:
                stats_df.to_csv(os.path.join(output_dir, f'{names[i]}_stats.csv'), index=False)
            
            # Create confusion matrix with optional cell type filtering
            cm, cell_types = create_confusion_matrix(merged_df, include_cell_types)
            
            # Plot confusion matrix
            print(f"Generating confusion matrix for {names[i]}...")
            fig_cm = plot_confusion_matrix(cm, cell_types, title=f'Confusion Matrix - {names[i]}')
            
            # Save confusion matrix plot
            if output_dir is not None:
                fig_cm.savefig(os.path.join(output_dir, f'{names[i]}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            
            # Calculate metrics for individual setting with optional cell type filtering
            metrics_df = calculate_metrics(merged_df, include_cell_types)
            
            # Save metrics
            if output_dir is not None:
                metrics_df.to_csv(os.path.join(output_dir, f'{names[i]}_metrics.csv'), index=False)
                
            # Plot metrics
            print(f"Generating metrics plot for {names[i]}...")
            fig_metrics = plot_metrics(metrics_df)
            
            # Save metrics plot
            if output_dir is not None:
                fig_metrics.savefig(os.path.join(output_dir, f'{names[i]}_metrics_plot.png'), dpi=300, bbox_inches='tight')
                
        except Exception as e:
            import traceback
            print(f"Error processing annotation setting {names[i]}: {str(e)}")
            print(traceback.format_exc())
            continue
    
    print("Analysis completed successfully!")
if __name__ == '__main__':
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cell Annotation Analysis')
    
    # Add arguments
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--annotations', required=True, nargs='+', help='Paths to annotation CSV files')
    parser.add_argument('--names', nargs='+', help='Names for annotation settings')
    parser.add_argument('--output-dir', default='results', help='Path to output directory')
    parser.add_argument('--include-cell-types', nargs='+', help='List of cell types to include in the analysis')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function
    main(args.ground_truth, args.annotations, args.names, args.output_dir, args.include_cell_types)
