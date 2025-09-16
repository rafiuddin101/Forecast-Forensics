import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ..reliability import reliability_mean, reliability_quantile, reliability_proba
from ..decomposition import decompose
from ..rstar import rstar
from ..plots.reliability_plot import plot_reliability
from ..plots.decomposition_plot import plot_decomposition

def main():
    parser = argparse.ArgumentParser(description='Forecast Forensics CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_rel = sub.add_parser('reliability', help='Compute reliability diagram data')
    p_rel.add_argument('--csv', required=True)
    p_rel.add_argument('--y', required=True)
    p_rel.add_argument('--pred', required=True)
    p_rel.add_argument('--functional', required=True, choices=['mean', 'quantile', 'proba'])
    p_rel.add_argument('--alpha', type=float, default=None)
    p_rel.add_argument('--n_bins', type=int, default=20)
    p_rel.add_argument('--plot', action='store_true', help='Generate reliability plot')
    p_rel.add_argument('--output', help='Save plot to file (e.g., reliability.png)')

    p_dec = sub.add_parser('decompose', help='Compute score decomposition + R*')
    p_dec.add_argument('--csv', required=True)
    p_dec.add_argument('--y', required=True)
    p_dec.add_argument('--pred', required=True)
    p_dec.add_argument('--functional', required=True, choices=['mean', 'median', 'quantile'])
    p_dec.add_argument('--alpha', type=float, default=None)
    p_dec.add_argument('--n_bins', type=int, default=20)
    p_dec.add_argument('--plot', action='store_true', help='Generate decomposition plot')
    p_dec.add_argument('--output', help='Save plot to file (e.g., decomposition.png)')

    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if args.cmd == 'reliability':
        if args.functional == 'mean':
            out = reliability_mean(df[args.y].values, df[args.pred].values, n_bins=args.n_bins)
            print(out.to_csv(index=False))
            if args.plot:
                fig = plot_reliability(out, 'x_pred_mean', 'y_empirical_mean', 
                                      title=f'Reliability Diagram (Mean, {args.n_bins} bins)')
                if args.output:
                    plt.savefig(args.output)
                plt.show()
        elif args.functional == 'proba':
            out = reliability_proba(df[args.y].values, df[args.pred].values, n_bins=args.n_bins)
            print(out.to_csv(index=False))
            if args.plot:
                fig = plot_reliability(out, 'x_pred_mean', 'y_empirical_mean', 
                                      title=f'Reliability Diagram (Probability, {args.n_bins} bins)')
                if args.output:
                    plt.savefig(args.output)
                plt.show()
        else:
            out = reliability_quantile(df[args.y].values, df[args.pred].values, alpha=args.alpha, n_bins=args.n_bins)
            print(out.to_csv(index=False))
            if args.plot:
                fig = plot_reliability(out, 'x_pred_mean', 'y_empirical_mean', 
                                      title=f'Reliability Diagram (Quantile {args.alpha}, {args.n_bins} bins)')
                if args.output:
                    plt.savefig(args.output)
                plt.show()

    elif args.cmd == 'decompose':
        dec = decompose(df[args.y].values, df[args.pred].values,
                        functional=args.functional, alpha=args.alpha, n_bins=args.n_bins)
        print(dec)
        r_star_value = rstar(dec)
        print('R* =', r_star_value)
        
        if args.plot:
            # Add R* to the decomposition result for plotting
            dec['rstar'] = r_star_value
            fig = plot_decomposition(dec, title=f'Score Decomposition ({args.functional})')
            if args.output:
                plt.savefig(args.output)
            plt.show()

if __name__ == '__main__':
    main()
