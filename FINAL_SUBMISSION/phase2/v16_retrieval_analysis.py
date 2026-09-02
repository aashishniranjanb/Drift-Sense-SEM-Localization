import os
import pandas as pd
import numpy as np

def main():
    print("Running V16 Retrieval Analysis...")
    audit_file = "results/v15/V15_ORACLE_AUDIT.csv"
    if not os.path.exists(audit_file):
        print(f"File not found: {audit_file}")
        return
        
    df = pd.read_csv(audit_file)
    
    # Filter for PRESENT cases
    df_present = df[df['raw_gt_available'] == 1].copy()
    
    print(f"Total present cases available in raw GT: {len(df_present)}")
    
    def assign_group(row):
        if row['candidate_top50'] == 1:
            return 'A'
        elif row['candidate_top500'] == 1:
            return 'B'
        else:
            return 'C'
            
    df_present['retrieval_group'] = df_present.apply(assign_group, axis=1)
    
    # Save the ceiling csv
    out_csv = "results/v16/retrieval_ceiling.csv"
    df_present.to_csv(out_csv, index=False)
    print(f"Saved retrieval ceiling to {out_csv}")
    
    # Generate MD report
    group_counts = df_present['retrieval_group'].value_counts()
    count_A = group_counts.get('A', 0)
    count_B = group_counts.get('B', 0)
    count_C = group_counts.get('C', 0)
    total = count_A + count_B + count_C
    
    report_path = "results/v16/V16_RETRIEVAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# V16 Retrieval Ceiling Report\n\n")
        f.write(f"Total present cases with raw GT available: {total}\n\n")
        f.write(f"**Group A (Top-50 Recovery):** {count_A} ({(count_A/total)*100:.2f}%)\n")
        f.write(f"**Group B (Rank 51-500 Recovery):** {count_B} ({(count_B/total)*100:.2f}%)\n")
        f.write(f"**Group C (>500 or Lost):** {count_C} ({(count_C/total)*100:.2f}%)\n\n")
        f.write("## Interpretation\n")
        f.write("The current NMS (r=5) limits the top 50, often filling it with periodic clones. ")
        f.write("Group B represents fully recoverable candidates that are pushed out of the top 50 slot capacity.\n")
        f.write("Next step: NMS experiments (R1, R2, R3) and periodic-family compression.\n")
        
    print(f"Saved report to {report_path}")

if __name__ == '__main__':
    main()
