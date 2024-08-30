import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


def load_and_prepare_data(file_paths):
    all_embeddings = []
    all_titles = []
    all_chunk_ids = []
    all_pdf_ids = []

    for file in file_paths:
        df = pd.read_csv(file)

        embeddings = np.array([np.fromstring(embedding.strip('[]'), sep=' ') for embedding in df['embeddings']])

        all_embeddings.append(embeddings)
        all_titles.extend(df['title'])
        all_chunk_ids.extend(df['chunk_id'])
        all_pdf_ids.extend(df['pdf_id'])

    combined_embeddings = np.vstack(all_embeddings)

    return combined_embeddings, all_titles, all_chunk_ids, all_pdf_ids


def main():
    csv_files = ['embeddings/DSC_embeddings_4518998.csv',
                 'embeddings/DSC_embeddings_6640040.csv']

    embeddings, titles, chunk_ids, pdf_ids = load_and_prepare_data(csv_files)

    # Apply t-SNE
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': titles,
        'chunk_id': chunk_ids,
        'pdf_id': pdf_ids
    })

    fig = px.scatter(df, x='x', y='y', color='title', marginal_y='histogram', hover_data=['title', 'pdf_id'],
                     opacity=0.7, title='t-SNE Visualisering av Chunks', template='simple_white')
    fig.show()


if __name__ == '__main__':
    main()
