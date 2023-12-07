from langchain.document_loaders import PyPDFLoader
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = chromadb.Client()

def create_embeddings(text_obj):
    chroma_text = PyPDFLoader(text_obj).load_and_split()
    list_of_chroma_result = [doc.page_content for doc in chroma_text]

    # For now we have dummy metadata TODO!
    metadatas = []
    ids = []
    num_objects = len(list_of_chroma_result)
    metadatas = [{"source": "my_source"} for _ in range(num_objects)]
    ids = ["id" + str(i + 1) for i in range(num_objects)]
    embedding = model.encode(list_of_chroma_result).tolist()
    return metadatas, ids, embedding, list_of_chroma_result

def create_or_get_collection(path):

    collection_name = "demo_docs"
    try:
        collection = client.get_collection(collection_name)
    except ValueError:
        embeddings_obj = create_embeddings(path)
        collection = client.create_collection(collection_name)
        collection.add(
            documents=embeddings_obj[3],
            embeddings=embeddings_obj[2],
            metadatas=embeddings_obj[0],
            ids=embeddings_obj[1]
        )
    return collection

def create_html(text):
    soup = BeautifulSoup(features="html.parser")

    head_tag = soup.new_tag('head')
    meta_tag1 = soup.new_tag('meta')
    meta_tag1.attrs['charset'] = "UTF-8"
    meta_tag2 = soup.new_tag('meta')
    meta_tag2.attrs['content'] = "IE=edge"
    meta_tag2.attrs['http-equiv'] = "X-UA-Compatible"
    meta_tag3 = soup.new_tag('meta')
    meta_tag3.attrs['content'] = "width=device-width, initial-scale=1.0"
    meta_tag3.attrs['name'] = "viewport"

    title_tag = soup.new_tag('title')
    title_tag.string = "User Guide"
    style_tag = soup.new_tag('style')
    style_tag.string = "h1 { font-size: 24px; } h2 { font-size: 20px; } ul { list-style-type: disc; } li { margin-bottom: 10px; }"
    head_tag.extend([meta_tag1, meta_tag2, meta_tag3, title_tag, style_tag])
    soup.append(head_tag)

    body_tag = soup.new_tag('body')
    soup.append(body_tag)
    lines = text.split('\n')
    for line in lines:
        if "User Guide EVOLUTION³" in line:
            h1_tag = soup.new_tag('h1')
            h1_tag.string = line.strip()
            body_tag.append(h1_tag)
        elif line.startswith(' '):
            ul_tag = soup.new_tag('ul')
            li_tag = soup.new_tag('li')
            li_tag.string = line.strip(' ▪ ')
            ul_tag.append(li_tag)
            body_tag.append(ul_tag)
        elif any(char.isdigit() for char in line):
            h2_tag = soup.new_tag('h2')
            h2_tag.string = line.strip()
            body_tag.append(h2_tag)
        else:
            p_tag = soup.new_tag('p')
            p_tag.string = line.strip()
            body_tag.append(p_tag)

    return str(soup)