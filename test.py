import rag
model = rag.load_model()
index = rag.load_index("artifacts/index")
#rag.answer_to_file("your question", index, model, "artifacts/context.txt")

state = 1
print("test")
while(state == 1):
    q = input("Query:")
    if(q == "end"):
        state = 0
        break
    rag.answer_to_file(q, index, model, "artifacts/context.txt")
