import rag
model = rag.load_model()
index = rag.load_index("artifacts/index")
#rag.answer_to_file("your question", index, model, "artifacts/context.txt")

state = 1
TOP_K = 5
WINDOW = 1
API_KEY = rag.prompt_for_key()
while(state == 1):
    q = input("Query:")
    if(q == "end"):
        state = 0
        break
    #rag.answer_to_file(q, index, model, "artifacts/context.txt")
    query_vector = rag.embed_query(q, model)
    hits = rag.retrieve(query_vector, index, k=TOP_K)
    expanded = rag.expand_neighbors(hits, index, window=WINDOW)
    context = rag.assemble_context(q, expanded)

    ans = rag.get_answer(q, context, API_KEY)
    print(ans)
