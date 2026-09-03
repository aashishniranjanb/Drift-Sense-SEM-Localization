import argparse
import pandas as pd
import numpy as np

def apply_safe_gate(input_csv, output_csv, threshold=0.843, mode='V25'):
    '''
    Safe Gate implementation:
    Operates strictly on V25 candidate predictions.
    Coordinates and pose parameters are 100% frozen.
    Only ound, score, and zero-coordinates on rejection are affected.
    '''
    df = pd.read_csv(input_csv)
    
    if mode == 'V25':
        # Pure untouched baseline decision
        pass
    elif mode == 'V27_GATE':
        df['found'] = (df['score'] > threshold).astype(int)
    elif mode == 'V27_COMBINED':
        df['found'] = (df['score'] > threshold).astype(int)
        
    # Enforce strict Phase 2 zero-coordinate rule
    mask_rej = (df['found'] == 0)
    df.loc[mask_rej, ['x', 'y', 'theta', 'scale']] = 0.0
    
    df.to_csv(output_csv, index=False)
    print(f'Safe gate applied ({mode}, threshold={threshold}). Saved to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.843)
    parser.add_argument('--mode', default='V25')
    args = parser.parse_args()
    apply_safe_gate(args.input, args.output, args.threshold, args.mode)
