import chromadb
print("chromadb imported", chromadb.__version__)
c = chromadb.EphemeralClient()
print("ephemeral client ok")
coll = c.create_collection("test")
print("collection created")
coll.add(ids=["1"], documents=["hello"], embeddings=[[0.1]*384])
print("add ok")
