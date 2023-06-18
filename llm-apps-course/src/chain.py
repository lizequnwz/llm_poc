"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging, os

from types import SimpleNamespace

#import wandb
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


def load_vector_store(config: SimpleNamespace, openai_api_key: str) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=config.vector_store_dir
    )

    return vector_store


def load_chain(config, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=config.model_name,
        temperature=config.chat_temperature,
        max_retries=config.max_fallback_retries,
    )
    qa_prompt = load_chat_prompt(f"{config.chat_prompt_dir}/chat_prompt.json")
    print(qa_prompt)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return qa_chain


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list,
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """
    result = chain(
        inputs={"question": question, "chat_history": chat_history},
        return_only_outputs=True,
    )
    response = f"Answer:\t{result['answer']}"
    return response