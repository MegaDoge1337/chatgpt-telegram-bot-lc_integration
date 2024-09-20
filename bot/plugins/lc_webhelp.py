import os
import logging
import codecs
import json
import pandas as pd
from typing import Dict
from .plugin import Plugin

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_transformers import (LongContextReorder)
from chromadb import PersistentClient


class LCWebHelpPlugin(Plugin):

    def __init__(self):
        self.vector_store_existed = os.path.isdir('../vector_store')
        self.vector_store_client = PersistentClient('../vector_store')
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
        self.documents_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=2000)
        self.reordering = LongContextReorder()
        self.prompt_template = """Изучи текст справки: 
    
        - {ctx}
    
        Ответь на вопрос: {qs}"""
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=['ctx', 'qs'])
        self.llm = ChatOpenAI(model_name='gpt-4', temperature=0.9, max_tokens=4000)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.vector_store = Chroma(embedding_function=self.embeddings, client=self.vector_store_client)

    def get_source_name(self) -> str:
        return 'Directum: WebHelp & Club'

    def get_spec(self) -> [Dict]:
        return [
            {
                "name": "webhelp_and_club",
                "description": "Searching in vector store with `WebHelp` (help, documentation) and `Directum Club` web-site of `DirectumRX` by `Directum`. ONLY - for questions about `DirectumRX` and `Directum`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to search"},
                    },
                    "required": ["query"],
                },
            }
        ]

    async def execute(self, function_name, **kwargs) -> Dict:
        query = kwargs.get('query')
        return self.search_in_webhelp(query)

    def migrate_vector_store(self) -> None:
        if self.vector_store_existed:
            logging.warning('Vector store already migrated, skipping')
            return None

        logging.info('Start migrating vector store')
        webhelp_raw = codecs.open('../docs/rx_webhelp.json', 'r', 'utf_8')
        webhelp = json.load(webhelp_raw)
        df = pd.DataFrame(webhelp)
        documents_loader = DataFrameLoader(df, page_content_column='content')
        documents = documents_loader.load()
        split_documents = self.documents_splitter.split_documents(documents)
        all_documents_count = len(split_documents)
        processed_documents_count = 0
        for document in documents:
            processed_documents_count += 1
            logging.info(f"[{processed_documents_count}/{all_documents_count}] {document}")
            temp_vector_store = Chroma.from_documents([document], embedding=self.embeddings,
                                                      client=self.vector_store_client,
                                                      persist_directory='../vector_store')
            temp_vector_store.persist()

        club_raw = codecs.open('../docs/rx_club.json', 'r', 'utf_8')
        club = json.load(club_raw)
        df = pd.DataFrame(club)
        documents_loader = DataFrameLoader(df, page_content_column='content')
        documents = documents_loader.load()
        split_documents = self.documents_splitter.split_documents(documents)
        all_documents_count = len(split_documents)
        processed_documents_count = 0
        for document in documents:
            processed_documents_count += 1
            logging.info(f"[{processed_documents_count}/{all_documents_count}] {document}")
            temp_vector_store = Chroma.from_documents([document], embedding=self.embeddings,
                                                      client=self.vector_store_client,
                                                      persist_directory='../vector_store')
            temp_vector_store.persist()
        logging.info('Finished migrating vector store')
        return None

    def search_in_webhelp(self, query: str) -> Dict:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        relevant = retriever.get_relevant_documents(query)
        reordering = self.reordering.transform_documents(relevant)
        list_context = []
        context_tokens_len = 0
        for document in reordering:
            document_tokens = self.text_splitter.count_tokens(text=document.page_content)
            if (context_tokens_len + document_tokens) <= 8000:
                context_tokens_len += document_tokens
                list_context.append(document)

        str_context = ''
        for document in list_context:
            str_context += document.page_content + '.'

        answer = self.chain.predict(ctx=str_context, qs=query)
        return {'answer': answer}
