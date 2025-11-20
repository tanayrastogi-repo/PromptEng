# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "langchain-core==1.0.7",
#     "openai==2.8.1",
#     "python-dotenv==1.2.1",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import os
    from types import SimpleNamespace

    ## Langchain related importbs
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableSequence
    from langchain_core.messages import HumanMessage, SystemMessage

    # OpenAI SDK to use OpenRouter
    from openai import OpenAI

    # Load the env variables
    from dotenv import load_dotenv

    load_dotenv()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # Master Prompt Engineering and LangChain PromptTemplates

    ## Objectives

    In this notebook, we lool at:

    - **Understand the basics of prompt engineering**: Gain a solid foundation in how to effectively communicate with LLM using prompts, setting the stage for more advanced techniques.

    - **Master advanced prompt techniques**: Learn and apply advanced prompt engineering methods such as few-shot and self-consistent learning to optimize the LLM's response.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    The exercise is setup to use the freely aviable **OpenAI GPT-OSS** model, accessed using Openrouter API. We use the following python packages to setup the code:

    *   [`openai`](https://github.com/openai/openai-python/tree/main): OpenAI SDK in Python that enables the use of LLMs. We use the [Openrouter API](https://openrouter.ai/openai/gpt-oss-20b:free/api) to provide us with the model access.
    *   [`langchain`](https://www.langchain.com/): Provides various chain and prompt functions from LangChain.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Openrouter API**
    You will need to create your own API to acess models. Check [documentation for creating API Key](https://openrouter.ai/docs/quickstart).
    """)
    return


@app.cell
def _():
    api = mo.ui.text(
        placeholder="API-KEY...", label="Openrouter API", kind="password"
    )
    api
    return (api,)


@app.cell
def _(api):
    def llm_reasoning_model(prompt_txt, reasoning=True):
        # Model from Openrouter
        model_id = "openai/gpt-oss-20b:free"  ## free model from OpenAI

        # Parameters
        ## REF: https://github.com/openai/openai-python/blob/main/src/openai/types/completion_create_params.py
        default_params = {
            "max_tokens": 1000,  # Kind of defines number of word generated
            "temperature": 1.0,  # between [0, 2].
        }
        default_params = SimpleNamespace(**default_params)

        # Set up credentials for OpenRouter
        base_url = "https://openrouter.ai/api/v1"
        api_key = api.value if api.value else os.environ["OPENROUTER_API"]

        # Setup the model
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt_txt}",
                }
            ],
            extra_body={"reasoning": {"enabled": reasoning}},
            max_tokens=default_params.max_tokens,
            temperature=default_params.temperature,
        )

        # Return the assistant message with reasoning_details
        return response.choices[0].message
    return (llm_reasoning_model,)


@app.cell
def _(llm_reasoning_model):
    def chat_model(messages):
        # User Query
        prompt = messages[-1].content

        # Getting model response
        response = llm_reasoning_model(prompt_txt=prompt)

        # Format the reasoning for better understanding
        reasons = [
            piece.strip()
            for piece in response.reasoning.split(".")
            if piece.strip()
        ]
        reasoning_text = "\n".join(f"- {chunk}" for chunk in reasons)
        # Out
        output = mo.md(f"""    
        {response.content} \n
        **Reasoning:**
        {reasoning_text}
        """)
        return output
    return (chat_model,)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 1. Basic prompt

    A **basic prompt** is the simplest form of prompting, where you provide a short text or phrase to the model without any special formatting or instructions. The model generates a continuation based on patterns it has learned during training. Basic prompts are useful for exploring the model's capabilities and understanding how it naturally responds to minimal input.

    {mo.icon("line-md:chat-bubble-filled")} Click on the Chat button to get some examples.
    """)
    return


@app.cell
def _(chat_model):
    mo.ui.chat(
        chat_model,
        prompts=[
            "The future of artificial intelligence is ",
            "The benefits of sustainable energy include ",
        ],
        show_configuration_controls=False,
    )
    return


@app.cell
def _():
    mo.md(rf"""
    ## 2. Zero-shot prompt
    [**Zero-shot prompting**](https://www.ibm.com/think/topics/zero-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) is a technique where the model performs a task without any examples or prior specific training on that task. This approach tests the model's ability to understand instructions and apply its knowledge to a new context without demonstration. Zero-shot prompts typically include clear instructions about what the model should do, allowing it to leverage its pre-trained knowledge effectively.

    Zero-shot learning is crucial for testing a model's ability to apply its pre-trained knowledge to new, unseen tasks without additional training. This capability is valuable for gauging the model's generalization skills.

    This approach helps understand how well the model can handle direct questions based on its underlying knowledge and reasoning abilities.

    {mo.icon("line-md:chat-bubble-filled")} Click on the Chat button to get some examples.
    """)
    return


@app.cell
def _(chat_model):
    mo.ui.chat(
        chat_model,
        prompts=[
            """
        Translate the following English phrase into Spanish.
        English: "I would like to order a coffee with milk and two sugars, please."
        """
        ],
        show_configuration_controls=False,
    )
    return


@app.cell
def _():
    mo.md(rf"""
    ## 3. One-shot prompt

    [**One-shot prompting**](https://www.ibm.com/think/topics/one-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) provides the model with a **single** example of the task before asking it to perform a similar task. This technique gives the model a pattern to follow, improving its understanding of the desired output format and style. One-shot learning is particularly useful when you want to guide the model's response without extensive examples.

    We provide an example of one-shot learning example where the model is given a single example to help guide key-word extraction from a sentance. 

    The prompt provides a sample keyword extraction pairing. This example serves as a guide for the model to understand the task context and desired format. The model is then tasked with new text without further guidance.

    {mo.icon("line-md:chat-bubble-filled")} Click on the Chat button to see the example in action.
    """)
    return


@app.cell
def _(chat_model):
    mo.ui.chat(
        chat_model,
        prompts=[
            """
        Here is an example of extracting keywords from a sentence:

        Sentence: "Cloud computing offers businesses flexibility, scalability, and cost-efficiency for their IT infrastructure needs."
        Keywords: cloud computing, flexibility, scalability, cost-efficiency, IT infrastructure. 


        ---
        Now, please extract the main keywords from the following sentence:

        Sentence: "Sustainable agriculture practices focus on biodiversity, soil health, water conservation, and reducing chemical inputs."
        Keywords:
        """
        ],
        show_configuration_controls=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 4. Few-shot prompt

    [**Few-shot prompting**](https://www.ibm.com/think/topics/few-shot-prompting?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-In-Context+Learning+and+Prompt+Templates-v3-GenAIcourse_1741386184) extends the one-shot approach by providing multiple examples (typically 2-5) before asking the model to perform the task. These examples establish a clearer pattern and context, helping the model better understand the expected output format, style, and reasoning. This technique is particularly effective for complex tasks where a single example might not convey all the nuances.

    We provide an example of few-shot learning by classifying emotions from text statements. 

    Let's provide the model with three examples, each labeled with an appropriate emotion—joy, frustration, and sadness—to establish a pattern or guideline on how to categorize emotions in statements.

    After presenting these examples, let's challenge the model with a new statement: "That movie was so scary I had to cover my eyes." The task for the model is to classify the emotion expressed in this new statement based on the learning from the provided examples. 

    {mo.icon("line-md:chat-bubble-filled")} Click on the Chat button to see the example in action.
    """)
    return


@app.cell
def _(chat_model):
    mo.ui.chat(
        chat_model,
        prompts=[
            """
    Here are few examples of classifying emotions in statements:

    Statement: 'I just won my first marathon!'
    Emotion: Joy

    Statement: 'I can't believe I lost my keys again.'
    Emotion: Frustration

    Statement: 'My best friend is moving to another country.'
    Emotion: Sadness


    Now, classify the emotion in the following statement:
    Statement: 'That movie was so scary I had to cover my eyes.’
    """
        ],
        show_configuration_controls=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 5. Self-consistency
    [**Self-consistency**](https://www.promptingguide.ai/techniques/consistency) is an advanced technique in which the model generates multiple independent solutions or answers to the same problem, then evaluates these different approaches to determine the most consistent or reliable result. This method enhances accuracy by leveraging the model's ability to approach problems from different angles and identify the most robust solution through comparison and verification.

    The provided example demonstrates the self-consistency technique by reasoning through multiple calculations for a single problem. The problem posed is: “When I was 6, my sister was half my age. Now I am 70, what age is my sister?”

    The prompt instructs, “Provide three independent calculations and explanations, then determine the most consistent result.” This encourages the model to engage in critical thinking and consistency checking, both of which are vital for complex decision-making processes.

    {mo.icon("line-md:chat-bubble-filled")} Click on the Chat button to see the example in action.
    """)
    return


@app.cell
def _(chat_model):
    mo.ui.chat(
        chat_model,
        prompts=[
            """
    When I was 6, my sister was half of my age. Now I am 70, what age is my sister?
    Provide three independent calculations and explanations, then determine the most consistent result.
    """
        ],
        show_configuration_controls=False,
    )
    return


if __name__ == "__main__":
    app.run()
