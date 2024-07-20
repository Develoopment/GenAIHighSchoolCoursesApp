import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from htmlTemplate import css, bot_template, user_template

# for RAGAS Evaluation
from ragas.metrics import (
    answer_correctness,
    faithfulness,
)

from datasets import Dataset


#this function uses PdfReaader from the PyPDF2 library to read the text from each pdf
#Note that this app can use multiple PDFs at the same time so we need a for loop
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs: #?What type is pdf - an object??
        pdf_reader = PdfReader(pdf) #initialize an PdfReader object on the current pdf document
        
        for page in pdf_reader.pages: # go through each page in this pdf and get the text and put in it the 'text' variable for later use (i.e. chunking)
            text += page.extract_text()
    return text

def get_text_chunks(text):
    #initializing the CharacterTextSplitter object from langchain framework
    #this will allow us to split the text we read from the pdfs into chunks so that token limit won't be exceeded when we pass this information to the LLMs
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #this is to ensure that there are no gaps where characters are missed
        length_function=len
    )

    #actually split it using the above defined configurations
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #get the text chunks, convert to embedding based on embedding confid (above line i.e. OpenAI Ada service)
    return vectorstore


# REALLY UNDERSTAND WHAT THIS FUNCTION DOES - IT IS THE CORE OF THE AI FUNCTIONALITY
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4", temperature=0.75)
    retriever = vectorstore.as_retriever()

    print(retriever.get_relevant_documents("Who is working on the project"))
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) # figure out how memory works in langchain (! this is important, this is what gets the response from the LLM)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= retriever, #! this is the line of code that inserts the embedding that was made from the text that is stored in the FAISS vector database as context when the LLM responds to user questions
        memory=memory,
    )


    # print("=======Type Below=========")
    # print(conversation_chain.return_source_documents)
    return conversation_chain

def get_conversation_chain_updated(vectorstore):
    llm = ChatOpenAI(model="gpt-4", temperature=0.75)
    retriever = vectorstore.as_retriever()

    prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

    prompt_get_answer = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\\n\\n{context}"), 
            MessagesPlaceholder(variable_name="chat_history"), 
            ("user","{input}")
        ]
    
    )

    document_chain=create_stuff_documents_chain(llm, prompt_get_answer)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = st.session_state.chat_history

    response = retrieval_chain.invoke({
        "chat_history":chat_history,
        "input":"Who is working on this project?"
    })
    print(response["answer"])


# Triggered when the user clicks submit
def handle_userinput(user_question):

    response = st.session_state.conversation({'question':user_question}) #this adds the user's question (and prompt) in the session state which then triggers the get_conversation_chain function to pass to LLM??
    st.session_state.chat_history = response['chat_history'] #chat_history is the memmory key (!Figure out what the means in addition to above funciton)

    print("+++++ " + user_question + " +++++")

    print("+++++ " + response["answer"] + " +++++")

    print("\nResult\n")
    # print(response)

    # The response['chat_history'] pulls up a json object with index 0, 2 etc are what the user typed in (the questions they asked)
    # the index of 1, 3, 5 are the responses the the LLM had replied with
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) #we are simply replacing the {{MSG}} part of the string in the HTML templates (check the HTMLtemplate file for more info)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    #Load Dotenv
    load_dotenv()

    #GUI
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    #if the conversation 'cookie' is not in session meaning the conversation chain is not in history due to previous click of process button
    #because streamlit apparently randomly reloads the UI
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.write(css, unsafe_allow_html=True) #loads the css file so that when we load in the html templates for the design of the user and bot text boxes the css is applied(user_template and bot_template respectively - check import statement)
    st.header("Pathfinder :books:")
    
    user_input = st.text_input("Ask a question about your courses:")
    if user_input:
        handle_userinput( user_input)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        #once the user clicks button to process the pdf (convert to chunks and then make them into embeddings)
        if st.button("Process"):
            with st.spinner("Processing"): #this tells the user graphically that there is a process running (with the spinning wheel)
                #get PDF text
                raw_text = get_pdf_text(pdf_docs)
                print("====================Pdf is being converted to text====================")
                #print(raw_text)

                #Make text chunks
                text_chunks = get_text_chunks(raw_text)
                print("====================Pdf is being split====================")

                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                print("====================PDf has been vectorized and now is stored in FAISS Vectorstore====================")

                #create conversion chain
                st.session_state.conversation = get_conversation_chain(vectorstore) #?Understand how session_state and persistent state mangement works in streamlit
                st.session_state.conversation = get_conversation_chain_updated(vectorstore)

        
        if st.button("RAGAS Test"):
            print("hello")

if __name__ == "__main__":
    main()
