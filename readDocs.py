from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from itertools import chain

load_dotenv()

# ---------------------------
# 1. Setup embeddings & vectorstore
# ---------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # cheap
)

vectorstore = Chroma(
    collection_name="docling_simple",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ---------------------------
# 2. Retrieve all documents metadata
# ---------------------------
data = vectorstore._collection.get(include=["metadatas"])

# Extract all roles (stored as joined string)
all_roles_nested = [
    meta.get("roles", "").split("|")
    for meta in data["metadatas"]
    if meta.get("roles")
]

# Flatten and deduplicate
unique_roles = sorted(set(chain.from_iterable(all_roles_nested)))

# ---------------------------
# 3. Print results
# ---------------------------
print(f"Found {len(unique_roles)} unique roles:")
for role in unique_roles:
    print("-", role)