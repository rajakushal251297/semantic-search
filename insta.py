import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import  streamlit as st

# use your dataset
data = pd.read_csv("questions.csv")


# use your api and environment

pinecone.init(api_key='4af90430-72dd-46d7-919a-63620484b003', environment='us-east-gcp')

model = SentenceTransformer('all-mpnet-base-v2',device='cuda')
embeding = model.encode("This is sentence")
pinecone.create_index(name='question-search', dimension=768)
index = pinecone.Index('question-search')

question_list = []
for i,row in data.iterrows():
  question_list.append(
      (
        str(row['id']),
        model.encode(row['question1']).tolist(),
        {
            'is_duplicate': int(row['is_duplicate']),
            'question1': row['question1']
        }
      )
  )
  if len(question_list)==50 or len(question_list)==len(data):
    index.upsert(vectors=question_list)
    question_list = []
    
st.title("Search here")
query=st.text_input("Enter your query")

if st.button("Search"):
    q = model.encode([query]).tolist()
    result = index.query(q, top_k=2, includeMetadata=True)
    for i in result["match"]:
        result=i["metadata"]["question1"]
        st.write(result)