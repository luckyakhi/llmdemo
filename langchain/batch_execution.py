from langchain.model_utils import init_and_get_model

if __name__ == '__main__':
    model = init_and_get_model()
    results = model.batch(["List top 5 symptoms of Anxiety attack","List top 5 symptoms of Panic attack"])
    for result in results:
        print(result.content)
        print('----------------')