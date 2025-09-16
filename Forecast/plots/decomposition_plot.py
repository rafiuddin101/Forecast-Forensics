import matplotlib.pyplot as plt
import numpy as np

def plot_decomposition(decomposition_result, title="Score Decomposition"):
    """
    Plot the decomposition of the score.
    
    Parameters:
    -----------
    decomposition_result : dict
        The result from the decompose function
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Extract values
    score = decomposition_result['score']
    unc = decomposition_result['UNC']
    dsc = decomposition_result['DSC']
    mcb = decomposition_result['MCB']
    rstar = decomposition_result.get('rstar', None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define components to plot
    components = ['Score', 'UNC', 'DSC', 'MCB']
    values = [score, unc, dsc, mcb]
    
    # Create bar plot
    bars = ax.bar(components, values, color=['blue', 'green', 'orange', 'red'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Add R* value if available
    if rstar is not None:
        ax.text(0.5, 0.95, f'R* = {rstar:.4f}', 
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    ax.set_ylabel('Value')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig