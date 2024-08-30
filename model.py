from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch

from preprocessing import preprocessing
from data_loading import read_pdf
from chunking import chunking


def tokenization(tokenizer, text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)


def main():
    model_name = 'NbAiLab/nb-bert-base'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pdf = '6640040'

    text = read_pdf(f'data/{pdf}.pdf')
    chunks_df = chunking(text)

    # chunks_df['title'] = chunks_df['title'].apply(preprocessing)
    chunks_df['content'] = chunks_df['content'].apply(preprocessing)

    all_embeddings = []
    chunk_titles = []
    chunk_ids = []

    for chunk_id, row in chunks_df.iterrows():
        tokenized_text = tokenization(tokenizer, row['content'])
        with torch.no_grad():
            output = model(**tokenized_text)
            embeddings = output.last_hidden_state.squeeze().numpy()
            all_embeddings.append(embeddings)
            chunk_titles.extend([row['title']] * embeddings.shape[0])
            chunk_ids.extend([chunk_id] * embeddings.shape[0])

    embeddings = np.vstack(all_embeddings)

    df = pd.DataFrame({
        'embeddings': [str(embedding) for embedding in embeddings],
        'title': chunk_titles,
        'chunk_id': chunk_ids,
        'pdf_id': [pdf] * len(chunk_titles)
    })

    df.to_csv(f'embeddings/DSC_embeddings_{pdf}.csv', index=False)


if __name__ == '__main__':
    main()