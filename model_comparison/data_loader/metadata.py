"""CSV paper metadata is not a user interaction source."""
import csv
import io
from pathlib import Path
from .schema import Paper, MetadataParseResult, DataError

def load_metadata(path, num_items, encoding='auto'):
    path=Path(path); raw=path.read_bytes(); warnings=[]
    try:
        if encoding=='auto':
            try: text=raw.decode('utf-8-sig'); used='utf-8-sig'
            except UnicodeDecodeError:
                text=raw.decode('latin-1'); used='latin-1'
                warnings.append('Strict UTF-8 failed; Latin-1 byte-preserving fallback does not establish original display encoding.')
        else: text=raw.decode(encoding); used=encoding
    except (UnicodeError,LookupError) as exc: raise DataError(f'{path}: metadata decoding: {exc}') from exc
    papers={}
    try:
        reader=csv.DictReader(io.StringIO(text,newline=''),strict=True)
        required={'doc.id','title','citeulike.id','raw.title','raw.abstract'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames): raise DataError(f'{path}: record 1: missing CSV headers')
        for record,row in enumerate(reader,2):
            try:
                if None in row or any(row[k] is None for k in required): raise ValueError('extra or missing CSV fields')
                doc=int(row['doc.id']); item=doc-1
                if not 0<=item<num_items or item in papers: raise ValueError('duplicate/out-of-range doc.id')
                papers[item]=Paper(item,doc,row['citeulike.id'] or None,row['title'] or None,row['raw.title'] or None,row['raw.abstract'] or None)
            except ValueError as exc: raise DataError(f'{path}: record {record}: {exc}') from exc
    except csv.Error as exc: raise DataError(f'{path}: record {reader.line_num}: {exc}') from exc
    if len(papers)!=num_items: raise DataError(f'{path}: incomplete metadata document mapping')
    return MetadataParseResult([papers[i] for i in range(num_items)],used,warnings)
