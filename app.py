from langchain.chains import RetrievalQA
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.llms import Ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
import chromadb
import os
import argparse
import time
import chainlit as cl
from typing import Dict, Optional

model = os.environ.get("MODEL", "llama2:13b")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful and Caring answer:
"""

def custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


@cl.on_chat_start
async def start():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)


    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents= True,
        chain_type_kwargs={'prompt': custom_prompt()})
    chain = qa
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt()
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

    cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain") 
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     #res = await chain(message)
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\n\nSources:" + str(sources)
#     else:
#         answer += "\n\nNo sources found"

#     await cl.Message(content=answer).send()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async with cl.Step(type="run", name="Doc Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  if provider_id == "google":
    if raw_user_data["hd"] == "innovaccer.com":
      return default_user
  return None





