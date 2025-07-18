import re
import csv
from datetime import datetime
import openpyxl

# Path to your log file
log_file_path = 'logs/audio_processing.log'
output_csv = 'transcription_analysis.csv'
output_xlsx = 'transcription_analysis.xlsx'

# Regex patterns
chunk_pattern = re.compile(r"AUDIO CHUNK (\d+) RECEIVED.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
saved_pattern = re.compile(r"AUDIO CHUNK (\d+) SAVED - File: (chunk_\d+.*?)\.wav")
added_to_queue_pattern = re.compile(r"Added chunk (\d+) to queue")
processing_pattern = re.compile(r"Processing chunk (\d+)")
processed_pattern = re.compile(r"Chunk (\d+) processed - Text: '(.*?)'")
transcribed_time_pattern = re.compile(r"chunk_(\d+)_.*?_.*?_(\d{8}_\d{6}_\d+)\.json")
sent_pattern = re.compile(r"Sent pending message: .*")

# Helper: extract datetime from line
def extract_time_from_log(line):
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
    return match.group(0) if match else ""

# Track data per chunk
chunks = {}

# Read log file
with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    # Audio received
    match = chunk_pattern.search(line)
    if match:
        chunk_id = match.group(1)
        ts = match.group(2)
        chunks[chunk_id] = {'audio_received': ts}

    # Saved as
    match = saved_pattern.search(line)
    if match:
        chunk_id, filename = match.groups()
        if chunk_id in chunks:
            chunks[chunk_id]['filename'] = filename + '.wav'

    # Added to queue
    match = added_to_queue_pattern.search(line)
    if match:
        chunk_id = match.group(1)
        if chunk_id in chunks:
            chunks[chunk_id]['added_to_queue'] = extract_time_from_log(lines[i])

    # Processing started
    match = processing_pattern.search(line)
    if match:
        chunk_id = match.group(1)
        if chunk_id in chunks:
            chunks[chunk_id]['processing_started'] = extract_time_from_log(lines[i])

    # Transcription completed
    match = processed_pattern.search(line)
    if match:
        chunk_id, text = match.groups()
        if chunk_id in chunks:
            chunks[chunk_id]['processing_finished'] = extract_time_from_log(lines[i])
            chunks[chunk_id]['transcription'] = text

    # Sent to UI
    match = sent_pattern.search(line)
    if match:
        # use last matched chunk
        for chunk_id in reversed(list(chunks.keys())):
            if 'sent_to_ui' not in chunks[chunk_id]:
                chunks[chunk_id]['sent_to_ui'] = extract_time_from_log(lines[i])
                break

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    header = [
        'Chunk ID', 'Audio Received', 'Saved As', 'Added to Queue',
        'Processing Started', 'Processing Finished', 'Duration (s)',
        'Transcription', 'Sent to UI'
    ]
    writer.writerow(header)

    for chunk_id, data in chunks.items():
        try:
            start = datetime.strptime(data['processing_started'], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(data['processing_finished'], "%Y-%m-%d %H:%M:%S")
            duration = (end - start).total_seconds()
        except:
            duration = ""

        writer.writerow([
            chunk_id,
            data.get('audio_received', ''),
            data.get('filename', ''),
            data.get('added_to_queue', ''),
            data.get('processing_started', ''),
            data.get('processing_finished', ''),
            duration,
            data.get('transcription', ''),
            data.get('sent_to_ui', ''),
        ])

# Write to Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Transcription Analysis"

# Write header
ws.append(header)

# Write data rows
for chunk_id, data in chunks.items():
    try:
        start = datetime.strptime(data['processing_started'], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(data['processing_finished'], "%Y-%m-%d %H:%M:%S")
        duration = (end - start).total_seconds()
    except:
        duration = ""

    ws.append([
        chunk_id,
        data.get('audio_received', ''),
        data.get('filename', ''),
        data.get('added_to_queue', ''),
        data.get('processing_started', ''),
        data.get('processing_finished', ''),
        duration,
        data.get('transcription', ''),
        data.get('sent_to_ui', ''),
    ])

wb.save(output_xlsx)
print(f"✅ Log analysis complete. Output saved to: {output_csv} and {output_xlsx}")
