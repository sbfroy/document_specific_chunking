import re
import pandas as pd
from data_loading import read_pdf


def chunking(text: str) -> pd.DataFrame:
    """
    Document Specific Chunking. Basing the chunks on where the "§" locates.
    :param text: str, the text to be chunked.
    :return: DataFrame, a df with section numbers, titles and content.
    """
    chunk_pattern = re.compile(r'§\s*(\d+(?:\.\d+)*)\s*(.*)')
    data = {
        'section': [],
        'title': [],
        'content': []
    }
    chunk_id, chunk_title = None, None
    chunk_content = []
    lines = text.split('\n')

    for line in lines:
        match = chunk_pattern.match(line)
        if match:
            if chunk_id is not None and len(' '.join(chunk_content).split()) >= 5:  # Only add to chunks if a prev exists.
                data['section'].append(chunk_id)
                data['title'].append(chunk_title)
                data['content'].append('\n'.join(chunk_content).strip())
            chunk_id = match.group(1)
            chunk_title = match.group(2).strip()
            chunk_content = []  # Reset chunk content.
        else:
            chunk_content.append(line)  # Append line to the current chunk.

    if chunk_id and len(' '.join(chunk_content).split()) >= 5:
        data['section'].append(chunk_id)
        data['title'].append(chunk_title)
        data['content'].append('\n'.join(chunk_content).strip())

    return pd.DataFrame(data)


def main():
    text = read_pdf('data/4518998.pdf')
    chunks_df = chunking(text)
    print(chunks_df)
    print(chunks_df.iloc[36]['section'])
    print(chunks_df.iloc[36]['title'])
    print(chunks_df.iloc[36]['content'])


if __name__ == '__main__':
    main()
