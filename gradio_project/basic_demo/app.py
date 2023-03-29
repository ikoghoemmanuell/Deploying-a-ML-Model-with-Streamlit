import gradio as gr


def greet(name):
    return "Hello " + name + "!!"


if __name__ == "__main__":
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch(
        # share=True,
    )
