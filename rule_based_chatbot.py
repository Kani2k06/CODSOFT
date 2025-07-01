def chatbot():
    print("Hi! I am a simple chatbot. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ").lower()

        if user_input == "hello" or user_input == "hi":
            print("Bot: Hello! How can I help you?")
        elif "your name" in user_input:
            print("Bot: I am a simple rule-based chatbot.")
        elif "how are you" in user_input:
            print("Bot: I'm just a program, but I'm functioning as expected!")
        elif "bye" in user_input or user_input == "exit":
            print("Bot: Goodbye! Have a great day!")
            break
        elif "help" in user_input:
            print("Bot: You can ask me about my name, greet me, or say bye to end.")
        else:
            print("Bot: I'm not sure how to respond to that. Try asking something else.")

if __name__ == "__main__":
    chatbot()
