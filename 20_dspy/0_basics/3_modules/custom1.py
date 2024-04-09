import dspy
from dspy import OllamaLocal

# ---------------------------------------------
# Local model
# ---------------------------------------------
model = OllamaLocal(model='mistral:v0.2')

# ---------------------------------------------
# DSPy configure
# ---------------------------------------------
dspy.configure(lm=model)

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

# Custom module
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_joke = dspy.Predict("theme -> joke")
        self.generate_story = dspy.Predict("joke -> story")
        self.generate_tweet = dspy.Predict("joke -> tweet")
        self.generate_email = dspy.Predict("story, tweet -> email")
        self.generate_answer = dspy.ChainOfThought("story, tweet -> answer", n=5)
    
    def forward(self, theme):
        joke = self.generate_joke(theme=theme).joke

        # parallel ?

        # serial
        story = self.generate_story(joke=joke).story
        tweet = self.generate_tweet(joke=joke).tweet

        # condition
        if(True):
            email = self.generate_email(story=story, tweet=tweet).email

        prediction = self.generate_answer(story=story, tweet=tweet).answer

        return dspy.Prediction(answer=prediction)


# RAG = Retreival Augmented Generation
qa = MyModule()

print("-------------------------------------------------------------")

theme = "software engineering on beach"
response = qa(theme)

print("---- theme ----")
print(theme)

print("---- Answer ----")
print(response.answer)

print("---- Full Response from model ----")
print(response)


print("--------------------------")
print("---- HISTORY OF STEPS ----")
print("--------------------------")
model.inspect_history(n=10)
