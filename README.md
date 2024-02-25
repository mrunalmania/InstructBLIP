
# InstructBLIP

## Description

This project implements a system for content description and embedding using various NLP techniques. It includes modules for generating embeddings from textual data, utilizing transformer-based models for content description, and integrating with Faiss for efficient similarity search.

## Files

- `app.py`: Contains the main application code using Gradio for building a user interface.
- `Content_description.py`: Implements content description functionality using InstructBlip models from the transformers library.
- `Creat_embedding.py`: Provides functionality for generating embeddings using SentenceTransformers and saving them to a pickle file.
- `description.csv`: Sample CSV file containing textual descriptions.
- `embeddings.pkl`: Pickle file containing pre-generated embeddings.

## Dependencies

- gradio
- pickle
- numpy
- faiss
- sentence_transformers
- torch
- transformers
- datasets
- pandas

## Usage

1. Ensure all dependencies are installed. You can install them using `pip install -r requirements.txt`.
2. Run `app.py` to start the application.
3. Use the interface to interact with the content description and embedding functionalities.

## Instructions

1. **Generating Embeddings**: Run `Creat_embedding.py` to generate embeddings from your text data and save them to a pickle file (`embeddings.pkl`).
2. **Content Description**: Utilize `Content_description.py` to describe content using pre-trained models from the transformers library.
3. **Application Interface**: Run `app.py` to launch a Gradio interface where you can interactively input text and perform content description and similarity search based on pre-generated embeddings.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Gradio](https://www.gradio.app/)
- [SentenceTransformers](https://www.sbert.net/)
- [Transformers](https://huggingface.co/transformers/)
- [Faiss](https://github.com/facebookresearch/faiss)
