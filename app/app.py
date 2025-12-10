from pathlib import Path
from typing import List

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "fine_tuned_model"
MAX_CHAR_LEN = 280
STYLE_TEMPS = {"Witty": 0.9, "Serious": 0.5, "Casual": 0.7}


@st.cache_resource
def load_generator(model_path: Path = MODEL_DIR):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def trim_tweet(text: str) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:MAX_CHAR_LEN]


def generate_tweets(
    generator,
    prompt: str,
    num_tweets: int,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    outputs = generator(
        prompt,
        num_return_sequences=num_tweets,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        return_full_text=False,
        eos_token_id=generator.tokenizer.eos_token_id,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [trim_tweet(o["generated_text"]) for o in outputs]


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest" not in st.session_state:
        st.session_state.latest = []


def main() -> None:
    init_state()
    st.title("AI Tweet Generator")
    st.write("Generate tweets in your style using the fine-tuned GPT-2 model.")

    prompt = st.text_input("Enter a tweet prompt:")
    style = st.selectbox("Choose tweet style:", list(STYLE_TEMPS.keys()))
    num_tweets = st.slider("Number of tweets", 1, 5, 3)
    max_new_tokens = st.slider("Max new tokens", 20, 120, 80, 5)
    top_k = st.slider("Top-k", 10, 200, 50, 5)
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.95, 0.01)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
    no_repeat_ngram_size = st.slider("No-repeat n-gram size", 1, 4, 2)
    seed = st.text_input("Optional seed (integer)", value="")

    if st.button("Generate Tweets"):
        prompt_clean = prompt.strip()
        if len(prompt_clean) < 3:
            st.warning("Please enter a longer prompt to generate tweets.")
            return

        if seed.strip():
            try:
                set_seed(int(seed.strip()))
            except ValueError:
                st.warning("Seed must be an integer. Using random seed.")

        try:
            generator = load_generator()
            with st.spinner("Generating tweets..."):
                tweets = generate_tweets(
                    generator=generator,
                    prompt=prompt_clean,
                    num_tweets=num_tweets,
                    temperature=STYLE_TEMPS[style],
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            st.session_state.latest = tweets

            for idx, tweet in enumerate(tweets, start=1):
                st.markdown(f"**Tweet {idx} ({style}):** {tweet}")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    if st.session_state.latest:
        if st.button("Save latest tweets to history"):
            st.session_state.history.extend(st.session_state.latest)
            st.success("Saved latest tweets to history.")

    st.subheader("Tweet History")
    if st.session_state.history:
        for idx, tweet in enumerate(st.session_state.history, start=1):
            st.markdown(f"{idx}. {tweet}")
        if st.button("Clear history"):
            st.session_state.history.clear()
            st.info("History cleared.")
    else:
        st.write("No tweets in history yet.")


if __name__ == "__main__":
    main()
