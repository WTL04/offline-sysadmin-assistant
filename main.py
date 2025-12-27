import os
from uuid import uuid4
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLSemanticPreservingSplitter,
)


def model_init(
    model_name="BAAI/bge-small-en",
    device="CPU",
    mean_pooling=True,
    normalize_embeddings=True,
):
    model_kwargs = {"device": device}
    encode_kwargs = {
        "mean_pooling": mean_pooling,
        "normalize_embeddings": normalize_embeddings,
    }

    ov_embeddings = OpenVINOEmbeddings(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )
    return ov_embeddings


def vectorstore_init(
    embedding_function,
    collection_name="arch_wiki_collection",
    persist_directory="chroma_langchain_db",
):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    return vector_store


def split_docs(docs, verbose=False):
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]

    # initilaize header splitter
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        max_chunk_size=3000,
        preserve_images=True,
        preserve_videos=True,
        elements_to_preserve=["table", "ul", "ol", "code"],
        denylist_tags=["script", "style", "head"],
    )

    # initilaize text splitter to ensure chunks fit in local llm
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    document_chunks = []

    for doc in docs:
        # if html file, split by header, then split again into smaller chunks
        if doc.metadata.get("source", "").endswith(".html"):
            semantic_splits = splitter.split_text(doc.page_content)

            # apply original source metadata to each split
            for split in semantic_splits:
                split.metadata["source"] = doc.metadata.get("source")

            # DEBUG

            if verbose:
                print(f"Found {len(semantic_splits)} sections.")
                if len(semantic_splits) > 0:
                    print(f"Metadata of first section: {semantic_splits[0].metadata}")

            document_chunks.extend(text_splitter.split_documents(semantic_splits))
        else:
            # non-html, use recursive splitter directly
            document_chunks.extend(text_splitter.split_documents([doc]))

    return document_chunks


def add_documents(vector_store, document_chunks, BATCH_SIZE=500, verbose=False):
    for i in range(0, len(document_chunks), BATCH_SIZE):
        # slice chunk list into batches
        batch = document_chunks[i : i + BATCH_SIZE]

        # generate random universally unique identifier (UUID)
        batch_uuids = [str(uuid4()) for _ in range(len(batch))]

        vector_store.add_documents(documents=batch, ids=batch_uuids)

        # DEBUG
        if verbose:
            print(
                f"Added batch {i // BATCH_SIZE + 1} (Documents {i} to {i + len(batch)})"
            )


def main():
    embed_model = model_init()
    vectorstore = vectorstore_init(embed_model)

    directory_path = "./arch-wiki/html/en/"

    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
    else:
        loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.html",  # all html files and subdirectories
            loader_cls=TextLoader,  # keeps html tags for document splitting
            show_progress=True,
        )

    docs = loader.load()  # loads all documents at once

    document_chunks = split_docs(docs)

    add_documents(vectorstore, document_chunks, verbose=True)


if __name__ == "__main__":
    main()
