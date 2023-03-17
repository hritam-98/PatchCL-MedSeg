import csv
import os
import sys

data = []

with open(sys.argv[1]) as info_file:
    reader = csv.reader(info_file)
    info = [(version, group) for version, group in reader if os.path.exists(f'logs/default/version_{version}')]

for version, group in info:
    with open(f'logs/default/version_{version}/metrics.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        data.extend(
            (
                (version, group, row['epoch'], row['best_val_mdc'])
                for row in reader
                if row['best_val_mdc']
            )
        )

with open('concat.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['file', 'group', 'epoch', 'dice'])
    writer.writerows(data)
