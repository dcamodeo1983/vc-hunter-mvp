
import csv

def merge_vc_sources(hardcoded_list, csv_file=None):
    vc_set = set(hardcoded_list)
    if csv_file:
        try:
            lines = csv_file.read().decode("utf-8").splitlines()
            reader = csv.reader(lines)
            for row in reader:
                if row:
                    vc_set.add(row[0].strip())
        except Exception as e:
            print(f"Failed to read CSV: {e}")
    return sorted(vc_set)
