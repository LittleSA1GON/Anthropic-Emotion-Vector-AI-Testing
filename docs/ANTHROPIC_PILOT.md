# Try the Anthropic-inspired pilot

Open http://127.0.0.1:8501 and choose **Setup**. Check and add the default `Qwen/Qwen2.5-0.5B-Instruct` model. Under **Built-in example**, choose **Anthropic research pilot**. It is loaded automatically; **Download example CSV** saves an editable copy. No upload is needed to run the built-in example.

Enter your Kaggle username. Leave the token blank to use your existing local Kaggle login, or enter your own API token. **Check account** verifies the credential. Click **Run experiment** when ready. This creates a private GPU job and uses your account's quota. Public Qwen weights need no Hugging Face token.

## What is in the CSV?

The file `data/templates/anthropic_emotion_pilot.csv` contains original synthetic prompts, not copied paper examples or measured results:

- 12 training pairs: four shared topics crossed with happy, calm and desperate. Each story is paired with a neutral account of the same topic.
- Six separate test stories: explicit and implicit scenarios for each of the three concepts, using new topics.
- One identical-input neutral control: its difference vector should be zero and its alignment unscorable, rather than an invented neutral probability.
- One speaker-conflict example: two characters suggest different emotions, so the expected label is deliberately blank. Inspect the scores and continuation without assigning a correctness grade.

The normal isolation option records negative and positive strengths for each vector on the first test. Together with the eight baseline test generations, this produces 14 generations for one model. Start with the default 32 output tokens, automatic layer selection and temperature zero.

Once Kaggle finishes, open **Results & steering** and click **Update results**. Select a case, inspect the input-minus-neutral gauges, and use **Experiment playback** to step through the captured states. For a fresh prompt, change one emotion slider first and compare the generated result with its zero baseline. Kaggle runs each new steering request as another batch, rather than streaming an interactive GPU session.

## Relationship to the paper

Source: Sofroniew et al. (2026), [Emotion Concepts and their Function in a Large Language Model](https://arxiv.org/html/2604.07729v1), Sections 1.1, 1.2 and 2.2.

The paper extracts emotion directions from many stories, pools activations over story tokens, centers across emotions, and removes dominant neutral-activation components. This pilot adapts the story contrasts to the existing lab: it uses matched neutral prompts and final-prefill-token measurements on your chosen open-weight checkpoint. The model, sample count, token pooling and denoising differ. It does not reproduce the paper or establish that a model experiences feelings.

Expected labels describe authored scenarios, not validated model predictions. Cosine scores remain exploratory and no thresholds are graded. The original 18-case course matrix remains available as a separate built-in example.
