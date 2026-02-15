
from pathlib import Path
from typing import List
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_postgres import PGVector

from dotenv import load_dotenv
load_dotenv()


# ---------------------------
# 1. Input folder
# ---------------------------
#input_folder = Path("docling_simple") 
input_folder = Path("doctags") 
doctags_files = list(input_folder.glob("*.doctags"))

if not doctags_files:
    print("No .doctags files found in input folder.")
    exit(1)


# ---------------------------
# 2. Define structured output schema
# ---------------------------
class ChunkInfo(BaseModel):
    chunk_id: str
    title: str
    content: str
    roles: List[str]
    keywords: List[str]

class ChunkingResult(BaseModel):
    chunks: List[ChunkInfo]


# ---------------------------
# 3. Prompt template
# ---------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
					You are an AI document analysis assistant.

                    Your task is to process the provided document (in DocTags format) and split it into meaningful, semantic chunks. 
                    Each chunk should represent a coherent piece of information that could be used later to answer user questions.

                    For each chunk, you must assign structured metadata including:

                    1. "chunk_id": a unique identifier for the chunk
                    2. "title": a descriptive title for the chunk
                    3. "content": the text of the chunk
                    4. "roles": Provide a list containing up to 3 distinct role names (minimum 1, maximum 3). Each role must represent the type of professional 
                    or department that would be responsible for answering questions related to this chunk in the future.
                    5. "keywords": a list of important keywords or concepts present in the chunk

                    Rules:

                    - Split the document into **semantic chunks** preserving meaning; do not invent content.
                    - Assign the **most appropriate roles** to each chunk based on its content.
                    - Be concise and accurate; each chunk should be independent and self-contained.
                    - Stay generic when selecting the roles, don't select sub-role like: Operations Management,Compliance. In this example Operations Management is enough. 
                    - Roles must be concise (2–4 words maximum).
                    - Use professional function titles, not personal names.
                    - Do not invent overly specific or fictional roles.
                    - Avoid generic labels like "Employee" or "Staff".
                    - Output must be a JSON array of strings.

                    The goal is to create a set of **structured chunks** ready for **embedding and vector storage**, including the role that will be used later for question answering.
                """
            ),
        ),
        (
            "human",
            (
                "Analyze the following document and produce semantic chunks.\n\n"
                "DOCUMENT:\n"
                "{doctags}"
            ),
        ),
    ]
)


# ---------------------------
# 4. LLM 
# ---------------------------
llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0
).with_structured_output(ChunkingResult)


# ---------------------------
# 5. Vector store setup
# ---------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
CONNECTION_STRING = "postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb"
collection_name = "tpg_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# ---------------------------
# 6. Process each file
# ---------------------------
for doctags_path in doctags_files:
    print(f"Processing {doctags_path.name}...")

    doctags_text = doctags_path.read_text(encoding="utf-8")

    # Run LLM chain
    chain = prompt | llm
    result = chain.invoke({"doctags": doctags_text})
    print(f'Processing LLM Result .... ')

    # Convert chunks to Document objects
    documents = []
    for chunk in result.chunks:
        doc = Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "roles": chunk.roles, 
                "keywords": chunk.keywords,
                "source": doctags_path.stem 
            }
        )
        print(f'Document: {doc.metadata}')
        documents.append(doc)

    # Add to vector store
    vector_store.add_documents(documents)
    print(f"Stored {len(documents)} chunks from {doctags_path.name}.")


print("All files processed.")
