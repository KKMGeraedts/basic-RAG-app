import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import download_loader

load_dotenv()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
) 
PDFReader = download_loader("PDFReader")

def get_index(data, index_name):
    """
    This creates an embedding for 'data' which should be an pdf document. The embedding
    used is called a vector store index. With the embedding the unstructured data: our pdf
    now has structure which can be used to look up in it or do other manipulations.
    """
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
            )
    
    return index

# Embed academic article
article_path = os.path.join("data", "academicpaper.pdf")
article = PDFReader().load_data(file=article_path)
article_index = get_index(article, "article")
article_engine = article_index.as_query_engine()

# Embed ESRS materiality assesment document
esrs_materiality_assessment_path = os.path.join("data", "MaterialityAssessment.pdf")
materiality_assessment = PDFReader().load_data(file=esrs_materiality_assessment_path)
materiality_assessment_index = get_index(materiality_assessment, "materiality_assessment")
materiality_assessment_engine = materiality_assessment_index.as_query_engine()