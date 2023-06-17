import csv

def convert_data(file_path, out_path):
    with open(file_path, 'r') as csv_file, open(out_path, 'w', newline='') as output_file:
        reader = csv.reader(csv_file)
        writer = csv.writer(output_file)

        next(reader, None)

        writer.writerow(['tokens', 'labels'])

        token_docs = []
        tag_docs = []

        current_tokens = []
        current_tags = []

        for row in reader:
            if len(row) < 4: # row increased by 1
                if current_tokens and current_tags:
                    token_docs.append(current_tokens)
                    tag_docs.append(current_tags)
                    writer.writerow([current_tokens, current_tags])

                    current_tokens = []
                    current_tags = []
            else:
                current_tokens.append(row[1])
                current_tags.append(row[2])

        if current_tokens and current_tags:
            token_docs.append(current_tokens)
            tag_docs.append(current_tags)
            writer.writerow([current_tokens, current_tags])

    return token_docs, tag_docs

convert_data('train.csv', 'transformed_train.csv')
convert_data('validation.csv', 'transformed_val.csv')

