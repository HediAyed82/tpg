from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from langchain_postgres import PGVector

import psycopg2
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()


class RoleInfo(BaseModel):
    role: str
    behaviour: str


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
CONNECTION_STRING = "postgresql://raguser:ragpass@postgres:5432/ragdb"
#"postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb"
collection_name = "tpg_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

answer_llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    streaming=True
)

MAX_HISTORY = 3
MAX_SYSTEM_PROMPT_CHARS = 1000


# ---------------------------
# 1. Find the relevant role
# ---------------------------
def find_role(question):
    # ---------------------------
    # 1. Load roles from Chroma
    # ---------------------------
    conn = psycopg2.connect(
        host="postgres",
        dbname="ragdb",
        user="raguser",
        password="ragpass",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT jsonb_array_elements_text(cmetadata->'roles') AS role
        FROM langchain_pg_embedding
        WHERE cmetadata ? 'roles'
        ORDER BY role;
    """)
    unique_roles = [row[0] for row in cur.fetchall()]
    allowed_roles_str = ", ".join(f"'{r}'" for r in unique_roles)
   
    cur.close()
    conn.close()

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
# 2. Extract chuncks from DB with similarity
# -------------------------------------------
def retrieve_docs(question, roleInfo: RoleInfo):
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "roles": [roleInfo.role]
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

    # Get session history
    history = cl.user_session.get("history", [])

    # Keep only last 3 exchanges (user+assistant pairs)
    history = history[-MAX_HISTORY * 2:]

    # Limit system prompt size
    limited_behaviour = roleInfo.behaviour[:MAX_SYSTEM_PROMPT_CHARS]

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
        MessagesPlaceholder(variable_name="history"),
        (
            "user",
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

    full_answer = ""

    async for chunk in chain.astream({
        "question": question,
        "behaviour": limited_behaviour,
        "role": roleInfo.role,
        "context": format_docs(docs),
        "history": history
    }):
        if chunk.content:
            full_answer += chunk.content
            await msg.stream_token(chunk.content)

    await msg.update()

    # ---------------------------
    # Save to session history
    # ---------------------------
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})

    # Trim again after append
    history = history[-MAX_HISTORY * 2:]

    cl.user_session.set("history", history)



@cl.on_chat_start
def start():
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):

    # 1️⃣ Detect role
    roleInfo = find_role(message.content)

    formatted_output = f"""
    ## 🎭 Selected Role
    **{roleInfo.role}**

    ## 🧠 Expected Behaviour
    {roleInfo.behaviour}
    """
    await cl.Message(content=formatted_output, author="Role Router").send()

    # 2️⃣ Retrieve docs
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
        await cl.Message(
            content="No relevant documents found.",
            author="Retriever"
        ).send()


    # 3️⃣ Display current session history (before generating new answer)
    history = cl.user_session.get("history", [])

    if history:
        formatted_history = "## 🧠 Current Session Memory\n\n"

        exchange_number = 1
        for i in range(0, len(history), 2):
            user_msg = history[i]["content"] if i < len(history) else ""
            assistant_msg = history[i+1]["content"] if i+1 < len(history) else ""

            formatted_history += (
                f"### 🔁 Exchange {exchange_number}\n"
                f"**👤 User:**\n{user_msg}\n\n"
                f"**🤖 Assistant:**\n{assistant_msg}\n\n"
            )

            exchange_number += 1

        await cl.Message(
            content=formatted_history,
            author="Session Memory"
        ).send()


    # 4️⃣ Ask LLM (history handled inside)
    await ask_question(message.content, docs, roleInfo)




@cl.password_auth_callback
def auth(username: str, password: str):
    if username == "hedi" and password == "hedi":
        return cl.User(identifier="hedi")
    if username == "lassad" and password == "lassad":
        return cl.User(identifier="lassad")
    if username == "damian" and password == "damian":
        return cl.User(identifier="damian")
    return None