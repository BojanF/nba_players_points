import csv


def parse_csv_file(file_path, range_from=1, range_to=82):
    results = []
    with open(file_path) as csvfile:
        data = list(csv.reader(csvfile, quoting=csv.QUOTE_ALL))
        if not (range_from == 1 and range_to == 82):
            data = data[range_from - 1: range_to]
        for row in data:  # each row is a list
            if row[4] == 'x':
                game_info = row[0:4]
                margin_and_odds = [None] * 3
                row = game_info + margin_and_odds
            results.append(row)
    return results

