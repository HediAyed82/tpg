from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from langchain_chroma import Chroma
import itertools
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()


class RoleInfo(BaseModel):
    role: str
    behaviour: str


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002" 
)

vectorstore = Chroma(
    collection_name="docling_simple",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

answer_llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    streaming=True
)


# ---------------------------
# 1. Find the relevant role
# ---------------------------
def find_role(question):

    # ---------------------------
    # 1. Load roles from Chroma
    # ---------------------------
    data = vectorstore._collection.get(include=["metadatas"])

    unique_roles = sorted({
        meta.get("role")
        for meta in data.get("metadatas", [])
        if meta and meta.get("role")
    })

    # Convert to string for prompt
    allowed_roles_str = ", ".join(f"'{r}'" for r in unique_roles)

    # ---------------------------
    # 2. LLM with structured output
    # ---------------------------
    llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0
    ).with_structured_output(RoleInfo)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                Find the most relevant role that can answer the user question.
                Also generate a behaviour that an agent having this role should have
                to respond to the user question.

                Allowed roles: {allowed_roles_str}

                Return only one role from the allowed list.
                """
            ),
            ("human", "{question}")
        ]
    )

    llm_chain = prompt | llm
    result = llm_chain.invoke({
        "question": question
    })

    return result


# -------------------------------------------
# 2. Extract chencks from DB with similarity
# -------------------------------------------
def retrieve_docs(question, roleInfo: RoleInfo):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5, 
            "filter": {
                "role": roleInfo.role 
            }
        }
    )
    
    return retriever.invoke(question)



#Add score threshold
#discard weak matches

#Multi-role fallback
# if no chunks for role → try secondary role


# -------------------------------------------
# 3. Answer user's question
# -------------------------------------------
def format_docs(docs):
    return "\n\n".join(
        f"""
            ### {doc.metadata.get('title', 'Untitled')}
            (Source: {doc.metadata.get('source')}, Chunk ID: {doc.metadata.get('chunk_id')})

            {doc.page_content}
            """.strip()
        for doc in docs
    )


async def ask_question(question, docs, roleInfo):

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are acting as a {role}.

            Behaviour guidelines:
            {behaviour}

            Rules:
            - Answer ONLY using the provided context
            - If the answer is not present, say "I do not have enough information"
            - Be precise, structured, and role-appropriate
            - Do not invent legal or technical interpretations
            """
        ),
        (
            "human",
            """
            User question:
            {question}

            Context:
            {context}
            """
        ),
    ])

    chain = answer_prompt | answer_llm

    msg = cl.Message(author=roleInfo.role, content="")
    await msg.send()

    async for chunk in chain.astream({
        "question": question,
        "behaviour": roleInfo.behaviour,
        "role": roleInfo.role,
        "context": format_docs(docs)
    }):
        await msg.stream_token(chunk.content)

    await msg.update()



@cl.on_message
async def main(message: cl.Message):
    # 1️⃣ Detect role from user question
    roleInfo = find_role(message.content)
    formatted_output = f"""
    ## 🎭 Selected Role
    **{roleInfo.role}**

    ## 🧠 Expected Behaviour
    {roleInfo.behaviour}
    """
    await cl.Message(content=formatted_output, author="Role Router").send()
    


    # 2️⃣ Retrieve relevant chunks from Chroma
    docs = retrieve_docs(message.content, roleInfo)
    if docs:
        formatted_sources = "## 📚 Retrieved Sources\n\n"

        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Untitled")
            source = doc.metadata.get("source", "Unknown source")

            formatted_sources += (
                f"### {i}. {title}\n"
                f"- 📄 **Source:** `{source}`\n\n"
            )

        await cl.Message(content=formatted_sources, author="Retriever").send()
    else:
        await cl.Message(content="No relevant documents found.", author="Retriever").send()


    
    # 3️⃣ Format context and answer question
    await ask_question(message.content, docs, roleInfo)