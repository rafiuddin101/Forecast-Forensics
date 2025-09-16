import matplotlib.pyplot as plt

def plot_reliability(df, x_col, y_col, title=None):
    plt.figure()
    plt.scatter(df[x_col], df[y_col])
    lo = min(df[x_col].min(), df[y_col].min())
    hi = max(df[x_col].max(), df[y_col].max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.tight_layout()
    return plt.gcf()
