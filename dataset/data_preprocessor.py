with open('./prices_filtered.csv') as data,\
        open('./prices_processed.csv', 'w') as out:
    header = True
    for l in data:
        if header:
            original_headers = l.strip().split(';')
            new_headers = original_headers[:12] + ['age', 'was_rebuilt'] + original_headers[14:]
            out.write(','.join(new_headers)+'\n')
            header = False
            continue

        row_data = l.strip().split(';')

        year = int(row_data[12])
        rec = int(row_data[13])

        new_data = row_data[:12] + [str(2019 - max(year, rec)), str(1 if rec > 0 else 0)] + row_data[14:]
        out.write(','.join(new_data) + '\n')
