import torch
import streamlit as st
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

@st.cache(allow_output_mutation=True)
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
st.title("요약 테스트(KoBART + AI hub)")
text = st.text_area("입력 (실행: command+enter)")

st.markdown("## 원문")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## 요약문")
    with st.spinner('처리중...'):
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(output)
