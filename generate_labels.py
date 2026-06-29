"""Generate a CSV of synthetic multi-labels for images in a directory.
Format: filename,prediabetes,insulin_resistance,hypertension
Labels are 0/1 generated deterministically with a seed.
"""
import argparse
import os
import random
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--out', default='labels.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    random.seed(args.seed)

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediabetes', 'insulin_resistance', 'hypertension'])
        for fn in files:
            # Generate correlated synthetic labels for demo purposes
            base = random.random()
            pre = 1 if base > 0.7 else 0
            ins = 1 if base > 0.6 else 0
            hyp = 1 if base > 0.8 else 0
            writer.writerow([fn, pre, ins, hyp])

    print('Wrote', args.out, 'with', len(files), 'rows')


if __name__ == '__main__':
    main()
