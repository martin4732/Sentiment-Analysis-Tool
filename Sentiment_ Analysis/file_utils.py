import PyPDF2
from preprocessing import preprocess_text

def read_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = file.read().decode('utf-8')
    return text

def load_and_preprocess(files):
    data = []
    for file in files:
        text = read_text(file)
        preprocessed_text = preprocess_text(text)
        data.append((text, preprocessed_text))
    return data
