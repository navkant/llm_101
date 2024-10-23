import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")

animal_type = st.sidebar.selectbox("what is your pet?", ("Cat", "Dog", "Hamster"))

match animal_type:
    case "Cat":
        pet_color = st.sidebar.text_area("What color is your cat?", max_chars=15)
    case "Dog":
        pet_color = st.sidebar.text_area("What color is your dog?", max_chars=15)
    case "Hamster":
        pet_color = st.sidebar.text_input("What color is your hamster?", max_chars=15)
    case _:
        pet_color = st.sidebar.text_input("What color is your pet?", max_chars=15)


if pet_color:
    response = lch.generate_pet_name(animal_type=animal_type, pet_color=pet_color)
    st.text(response["pet_name"])





