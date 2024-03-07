
files = ["Podcast/jre-1169-elon-musk", "Podcast/jre-1309-naval-ravikant", "Podcast/jre-1470-elon-musk", "Standup/joe-rogan-Strange-times-2018", "Standup/joe-rogan-Triggered-2016"]

for filename in files:
    # Open the file in read mode and read the content
    with open(f'{filename}.txt', 'r') as file:
        content = file.read()

    # Remove tabs and new lines
    content_without_tabs_newlines = content.replace("\t", "").replace("\n", " ")

    # Open the file in write mode and write the content without tabs and new lines
    with open(f'{filename}-formatted.txt', 'w') as output_file:
        output_file.write(content_without_tabs_newlines)
