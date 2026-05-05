import pandas as pd
from pathlib import Path


def generate_report(results_path: str = "evaluation/results.csv", output_path: str = "evaluation/report.md"):
    """Generate a markdown evaluation report from results CSV."""
    df = pd.read_csv(results_path)

    report = "# Passport Assistant — Evaluation Report\n\n"
    report += "## Overall Metrics\n\n"
    report += f"| Metric | Value |\n|---|---|\n"
    report += f"| Mean Correctness | {df['correctness'].mean():.3f} |\n"
    report += f"| Mean Faithfulness | {df['faithfulness'].mean():.3f} |\n"
    report += f"| Mean Latency | {df['latency_s'].mean():.2f}s |\n"
    report += f"| High Confidence % | {(df['confidence']=='high').mean()*100:.1f}% |\n"
    report += f"| Total Questions | {len(df)} |\n\n"

    report += "## Results by Category\n\n"
    report += "| Category | Correctness | Faithfulness | Count |\n|---|---|---|---|\n"
    for cat, group in df.groupby("category"):
        report += f"| {cat} | {group['correctness'].mean():.3f} | {group['faithfulness'].mean():.3f} | {len(group)} |\n"

    report += "\n## Detailed Results\n\n"
    for _, row in df.iterrows():
        report += f"### Q: {row['question']}\n"
        report += f"- **Expected:** {row['expected'][:200]}\n"
        report += f"- **Generated:** {row['generated'][:200]}\n"
        report += f"- **Correctness:** {row['correctness']:.3f} | **Faithfulness:** {row['faithfulness']:.3f}\n\n"

    Path(output_path).write_text(report, encoding="utf-8")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
