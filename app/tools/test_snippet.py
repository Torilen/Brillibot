import openai
import sys

if __name__ == '__main__':
    openai.api_key = sys.argv[1]

    response = openai.Completion.create(
        engine="davinci",
        prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
                               "\n"+''.join(['\nHuman: Hi how are you ?'])+"\nAI:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    print("TEXT:"+response["choices"][0]["text"][1:])