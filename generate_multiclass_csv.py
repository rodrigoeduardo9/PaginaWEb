import os
import csv
import argparse


def main(root_dir, out='multiclass_labels.csv'):
    rows = []
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fn in sorted(os.listdir(label_dir)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                rows.append([os.path.join(label_dir, fn), label])
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath','label'])
        writer.writerows(rows)
    print('Wrote', out, 'with', len(rows), 'rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default='multiclass_labels.csv')
    args = parser.parse_args()
    main(args.root, args.out)
