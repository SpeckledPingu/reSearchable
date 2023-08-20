from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import json
import os

class ImagePDFExtractor():
    def __int__(self, path, extension = None, destination='.'):
        self.path = Path(path)
        self.destination_path = Path(destination)
        self.extension = extension if extension is not None else '*'
        self.files = sorted(list(self.path.glob(f'*.{self.extension}')))
        self.model = ocr_predictor(pretrained=True)

    def _extract_pdf(self, file):
        doc = DocumentFile.from_pdf(file)
        result = self.model(doc)
        pages = result.export()['pages']
        text = defaultdict(list)
        for page in pages:
            for block in page['blocks']:
                line = ' '.join([x['value'] for x in block['lines'][0]['words']])
                text[page['page_idx']].append(line)

        for page_idx, _text in text.items():
            text[page_idx] = ' '.join(_text)
        return text

    def extract_folder(self):
        for path in tqdm(self.files):
            doc = self._extract_pdf(path)
            with open(self.destination_path.joinpath(f'extracted_{path.name}'), 'w') as f:
                json.dump(doc, f)

class TextPDFExtractor():
    def __int__(self):
        pass